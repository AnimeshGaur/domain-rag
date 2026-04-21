"""
retrieval/query_engine.py
──────────────────────────
Query orchestration layer — Phase 1 (search) and Phase 2 (answer).

Enhancement stack (all gracefully degrade when LLM is None):
  1. HyDE — embed a hypothetical answer instead of the raw query (better kNN recall)
  2. Multi-Query Expansion — rephrase into N variants and cross-query RRF merge
  3. Parent-Child Context — expand top results with full parent_text before LLM

search(question, filters) → SearchResponse     [NO LLM required — Phase 1]
answer(question, filters, stream)              [LLM required — Phase 2]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

from config import cfg
from embedding.base import get_embed_provider
from search.hybrid_search import hybrid_search, hits_to_results
from search.reranker import rerank
from processing.doc_schema import SearchResult

log = logging.getLogger(__name__)

# Lazy-init singletons (shared across requests)
_embed_provider = None


def _get_embed():
    global _embed_provider
    if _embed_provider is None:
        _embed_provider = get_embed_provider()
    return _embed_provider


def _get_llm_or_none():
    """Return the configured LLM provider, or None if not available (Phase 1)."""
    try:
        from llm.base import get_llm_provider
        return get_llm_provider()
    except Exception:
        return None


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SearchResponse:
    """Returned by search() — no LLM involved."""
    question: str
    results: list[SearchResult]
    retrieval_hits: int
    reranked_to: int


# ── Public: Search (Phase 1, no LLM) ─────────────────────────────────────────

def search(
    question: str,
    filters: dict | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
) -> SearchResponse:
    """
    Semantic + keyword hybrid search with cross-encoder reranking.
    Applies all enabled enhancements: HyDE, Multi-Query Expansion.
    No LLM required for Phase 1 — all LLM features degrade gracefully.

    Enhancement pipeline:
      1. Attempt HyDE (generate hypothetical answer → embed that instead)
      2. Attempt Multi-Query Expansion (rephrase → 3× search → cross-RRF)
      3. Rerank candidates with cross-encoder

    Args:
        question:     Natural language query.
        filters:      Optional metadata filters.
        top_k:        Number of candidates from hybrid search.
        rerank_top_n: Number of results after reranking.

    Returns:
        SearchResponse with ranked SearchResult list.
    """
    k = top_k or cfg.RETRIEVAL_TOP_K_DENSE
    n = rerank_top_n or cfg.RETRIEVAL_RERANK_TOP_N

    log.info("search() | question=%r | filters=%s", question[:80], filters)

    embed = _get_embed()
    llm = _get_llm_or_none()

    # ── Enhancement 1: HyDE ───────────────────────────────────────────────────
    # Generate a hypothetical answer and embed that instead of the raw question.
    # Falls back to standard embed_query if LLM is unavailable.
    if llm is not None:
        log.debug("HyDE: generating hypothetical answer for query")
        from search.hyde import hyde_embed
        q_vec = hyde_embed(question, llm, embed)
    else:
        q_vec = embed.embed_query(question)

    # ── Enhancement 2: Multi-Query Expansion ──────────────────────────────────
    # Rephrase query into N variants, search each, merge via cross-query RRF.
    # Falls back to single hybrid search if LLM is unavailable.
    if llm is not None:
        log.debug("Multi-Query: expanding query into variants")
        from search.query_expansion import expand_query, multi_query_search
        variants = expand_query(question, llm, n=3)
        hits = multi_query_search(
            variants=variants,
            embed_provider=embed,
            filters=filters,
            top_k=k,
        )
    else:
        hits = hybrid_search(
            query_vec=q_vec,
            query_text=question,
            top_k=k,
            filters=filters,
        )

    retrieval_hits = len(hits)
    log.info("Retrieval: %d hits (HyDE=%s, MultiQuery=%s)", retrieval_hits, llm is not None, llm is not None)

    # ── Rerank ────────────────────────────────────────────────────────────────
    reranked_hits = rerank(question, hits, top_n=n)
    results = [SearchResult.from_hit(h, rerank_score=h.get("rerank_score")) for h in reranked_hits]

    log.info("Reranked: %d results", len(results))
    return SearchResponse(
        question=question,
        results=results,
        retrieval_hits=retrieval_hits,
        reranked_to=len(results),
    )


# ── Public: Answer (Phase 2, requires LLM) ───────────────────────────────────

def answer(
    question: str,
    filters: dict | None = None,
    stream: bool = False,
):
    """
    Full RAG answer: search() + parent context expansion + LLM synthesis.
    Raises LLMNotEnabledError if LLM is not configured.

    Enhancement 3 — Parent-Child Context:
      After reranking, each result's text is expanded with its parent_text
      (the full section) before being passed to the LLM. This gives the LLM
      rich context even when search matched a small child chunk.

    Args:
        question: Natural language question.
        filters:  Optional metadata filters.
        stream:   If True, returns a token Iterator. If False, returns AnswerResult.
    """
    from llm.base import get_llm_provider, LLMNotEnabledError
    from llm.answer_engine import build_answer, stream_answer

    llm = get_llm_provider()  # raises LLMNotEnabledError if not configured

    search_resp = search(question, filters)

    # ── Enhancement 3: Parent-Child Context Expansion ─────────────────────────
    # For each result that has a parent_text, replace its text with the full
    # parent section before sending to the LLM. This avoids truncated context.
    expanded_results = _expand_with_parent_context(search_resp.results)

    if stream:
        return stream_answer(
            question=question,
            results=expanded_results,
            llm=llm,
            retrieval_hits=search_resp.retrieval_hits,
        )

    return build_answer(
        question=question,
        results=expanded_results,
        llm=llm,
        retrieval_hits=search_resp.retrieval_hits,
    )


# ── Helper: parent context expansion ─────────────────────────────────────────

def _expand_with_parent_context(results: list[SearchResult]) -> list[SearchResult]:
    """
    For each SearchResult that has a non-empty parent_text, build an expanded
    copy where `.text` is replaced by the full parent section text.

    This gives the LLM the full document section rather than the small child
    chunk that matched during retrieval — significantly improving answer quality.
    """
    expanded = []
    for r in results:
        src = r  # default: use as-is
        # parent_text is stored in the ES _source
        # SearchResult.from_hit reads it if present
        if getattr(r, "parent_text", "") and len(r.parent_text) > len(r.text):
            log.debug(
                "Parent expand: chunk '%s' %d→%d chars",
                r.chunk_id, len(r.text), len(r.parent_text),
            )
            # Replace text with the richer parent section
            import dataclasses
            src = dataclasses.replace(r, text=r.parent_text)
        expanded.append(src)
    return expanded
