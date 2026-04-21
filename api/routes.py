"""
api/routes.py
──────────────
FastAPI routes for RAGbase.

Endpoints:
  POST /api/ingest          — Trigger ingestion pipeline (background task)
  GET  /api/status          — Index stats + pipeline state
  POST /api/search          — Semantic+keyword search (NO LLM required)
  POST /api/query           — Full RAG answer (LLM required)
  POST /api/query/stream    — Streaming RAG answer (LLM required)
  GET  /api/sources         — Facets: doc_types, categories, tags, repos
  DELETE /api/index         — Drop the Elasticsearch index
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import cfg
from ingestion.pipeline import run_ingestion, get_ingestion_state
from retrieval.query_engine import search, answer
from search.elastic_store import index_stats, list_facets, delete_index
from llm.base import LLMNotEnabledError

log = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class IngestRequest(BaseModel):
    repos: list[str] = Field(default_factory=list, description="Owner/repo list")
    full_reset: bool = Field(False, description="Drop index before ingesting")
    use_llm: Optional[bool] = Field(None, description="Override INGEST_USE_LLM setting")
    artifact_types: Optional[list[str]] = Field(None, description="Override artifact types for LLM mode")


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Search query")
    filters: Optional[dict] = Field(None, description="Metadata filters, e.g. {doc_type: api_ref}")
    top_k: int = Field(20, ge=1, le=100, description="Candidate pool size")
    rerank_top_n: int = Field(5, ge=1, le=50, description="Results after reranking")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Natural language question")
    filters: Optional[dict] = Field(None, description="Metadata filters")
    top_k: int = Field(20, ge=1, le=100)
    rerank_top_n: int = Field(5, ge=1, le=20)


class SearchResultOut(BaseModel):
    chunk_id: str
    title: str
    description: str
    text: str
    score: float
    rerank_score: Optional[float]
    doc_type: str
    category: str
    artifact_type: str
    tags: list[str]
    repo: str
    url: str
    heading_path: str
    image_refs: list[str]


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResultOut]
    retrieval_hits: int
    reranked_to: int


class AnswerSource(BaseModel):
    title: str
    artifact_type: str
    doc_type: str
    category: str
    repo: str
    url: str
    score: float
    image_refs: list[str]


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[AnswerSource]
    retrieval_hits: int
    metadata: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest", summary="Trigger ingestion pipeline")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger the ingestion pipeline in the background.
    Poll /status to check progress.

    Phase 1 (no LLM): Set use_llm=false (or leave unset with INGEST_USE_LLM=false)
    Phase 2 (LLM):    Set use_llm=true (requires OPENAI_API_KEY)
    """
    state = get_ingestion_state()
    if state.running:
        raise HTTPException(409, "Ingestion pipeline is already running.")

    background_tasks.add_task(
        run_ingestion,
        repos=req.repos or None,
        full_reset=req.full_reset,
        use_llm=req.use_llm,
        artifact_types=req.artifact_types,
    )
    return {
        "status": "started",
        "mode": "llm" if req.use_llm else "no-llm",
        "repos": req.repos or cfg.GITHUB_REPOS,
    }


@router.get("/status", summary="Index stats and pipeline state")
async def status():
    """Returns Elasticsearch index stats and current ingestion pipeline state."""
    state = get_ingestion_state()
    es_stats = index_stats()
    return {
        "ingestion": state.to_dict(),
        "es_index": es_stats,
        "config": {
            "embed_provider": cfg.EMBED_PROVIDER,
            "embed_model": cfg.OPENAI_EMBED_MODEL if cfg.EMBED_PROVIDER == "openai" else cfg.LOCAL_EMBED_MODEL,
            "embed_dims": cfg.embed_dims,
            "llm_enabled": cfg.llm_enabled,
            "llm_provider": cfg.LLM_PROVIDER,
            "llm_model": (
                cfg.OPENAI_CHAT_MODEL if cfg.LLM_PROVIDER == "openai" 
                else cfg.GROQ_CHAT_MODEL if cfg.LLM_PROVIDER == "groq" 
                else None
            ) if cfg.llm_enabled else None,
            "index": cfg.ES_INDEX,
        },
    }


@router.post("/search", response_model=SearchResponse, summary="Semantic + keyword search (no LLM)")
async def search_endpoint(req: SearchRequest):
    """
    Hybrid semantic + keyword search with cross-encoder reranking.

    **No LLM required** — works in Phase 1 with only GITHUB_TOKEN.

    Filters available: doc_type, category, artifact_type, tags, repo, section
    """
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search(
                question=req.question,
                filters=req.filters,
                top_k=req.top_k,
                rerank_top_n=req.rerank_top_n,
            ),
        )
    except Exception as exc:
        log.exception("Search failed: %s", exc)
        raise HTTPException(500, f"Search error: {exc}")

    results_out = [
        SearchResultOut(
            chunk_id=r.chunk_id,
            title=r.title,
            description=r.description,
            text=r.text[:500],  # truncate for API response
            score=round(r.score, 4),
            rerank_score=round(r.rerank_score, 4) if r.rerank_score else None,
            doc_type=r.doc_type,
            category=r.category,
            artifact_type=r.artifact_type,
            tags=r.tags,
            repo=r.repo,
            url=r.url,
            heading_path=r.heading_path,
            image_refs=r.image_refs,
        )
        for r in resp.results
    ]

    return SearchResponse(
        question=resp.question,
        results=results_out,
        retrieval_hits=resp.retrieval_hits,
        reranked_to=resp.reranked_to,
    )


@router.post("/query", response_model=QueryResponse, summary="Full RAG answer (LLM required)")
async def query_endpoint(req: QueryRequest):
    """
    Semantic search + LLM-grounded answer with citations.

    **Requires** LLM_PROVIDER=openai and OPENAI_API_KEY to be set.
    Use /search if you only have Phase 1 set up.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: answer(question=req.question, filters=req.filters),
        )
    except LLMNotEnabledError as exc:
        raise HTTPException(501, str(exc))
    except Exception as exc:
        log.exception("Query failed: %s", exc)
        raise HTTPException(500, f"Query error: {exc}")

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            AnswerSource(
                title=s.title,
                artifact_type=s.artifact_type,
                doc_type=s.doc_type,
                category=s.category,
                repo=s.repo,
                url=s.url,
                score=s.score,
                image_refs=s.image_refs,
            )
            for s in result.sources
        ],
        retrieval_hits=result.retrieval_hits,
        metadata=result.metadata,
    )


@router.post("/query/stream", summary="Streaming RAG answer (LLM required)")
async def query_stream_endpoint(req: QueryRequest):
    """
    SSE streaming version of /query.
    Yields server-sent events with JSON payloads: `data: {"token": "..."}\\n\\n`
    The final event contains a metadata sentinel with sources.
    """
    try:
        token_iter = answer(question=req.question, filters=req.filters, stream=True)
    except LLMNotEnabledError as exc:
        raise HTTPException(501, str(exc))
    except Exception as exc:
        log.exception("Stream init failed: %s", exc)
        raise HTTPException(500, f"Stream error: {exc}")

    import json

    async def event_generator():
        loop = asyncio.get_event_loop()
        # Run blocking iterator in thread pool
        def _collect():
            return list(token_iter)

        tokens = await loop.run_in_executor(None, _collect)
        for token in tokens:
            payload = json.dumps({"token": token})
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sources", summary="Available facets (doc types, categories, repos)")
async def list_sources():
    """
    Returns facets from the current index:
    doc_types, categories, artifact_types, tags, repos.
    Use these values to build filter dropdowns in the UI.
    """
    try:
        return list_facets()
    except Exception as exc:
        raise HTTPException(500, f"Facet error: {exc}")


@router.delete("/index", summary="Drop the Elasticsearch index")
async def drop_index():
    """
    Permanently deletes the Elasticsearch index.
    Use with caution — all indexed data will be lost.
    Re-run /ingest to rebuild.
    """
    try:
        delete_index()
        return {"status": "deleted", "index": cfg.ES_INDEX}
    except Exception as exc:
        raise HTTPException(500, f"Delete error: {exc}")
