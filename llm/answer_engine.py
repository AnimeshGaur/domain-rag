"""
llm/answer_engine.py
─────────────────────
Grounded answer synthesis using an LLM.

Takes re-ranked search results and a user question,
assembles a grounded prompt, and returns a streamed or
full answer with [Source N] citations.

The LLM provider is injected — no hardcoded OpenAI dependency.
"""
from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Iterator

from processing.doc_schema import SearchResult
from llm.base import LLMProvider

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a knowledgeable assistant with access to a technical documentation knowledge base.
    The knowledge base covers software architecture, APIs, business processes, and user guides
    from a Docusaurus-hosted documentation site.

    Rules:
    — Answer ONLY using information from the provided context chunks.
    — Be precise and technical. When relevant, include code examples from the context.
    — Cite every factual claim with [Source N] where N is the chunk number (1-indexed).
    — If the context does not contain enough information, say so clearly — do not hallucinate.
    — Format your answer in Markdown with headings and bullet points where it improves clarity.
    — For multi-step processes, use numbered lists.
    — For API info, use tables or code blocks.
    — Keep answers focused and concise.
""").strip()


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AnswerSource:
    title: str
    artifact_type: str
    doc_type: str
    category: str
    repo: str
    url: str
    heading_path: str
    score: float
    image_refs: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[AnswerSource]
    retrieval_hits: int
    metadata: dict = field(default_factory=dict)


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _build_prompt(question: str, results: list[SearchResult]) -> list[dict]:
    context_parts: list[str] = []
    for i, r in enumerate(results, 1):
        meta_line = " | ".join(filter(None, [
            r.artifact_type or r.doc_type,
            r.category,
            r.repo,
            r.heading_path,
        ]))
        context_parts.append(
            f"[Source {i}] ({meta_line})\n"
            f"Title: {r.title}\n"
            f"---\n{r.text}\n"
        )

    context_text = "\n\n".join(context_parts)
    user_content = f"Context:\n{context_text}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _results_to_sources(results: list[SearchResult]) -> list[AnswerSource]:
    return [
        AnswerSource(
            title=r.title,
            artifact_type=r.artifact_type or r.doc_type,
            doc_type=r.doc_type,
            category=r.category,
            repo=r.repo,
            url=r.url,
            heading_path=r.heading_path,
            score=round(r.rerank_score or r.score, 4),
            image_refs=r.image_refs,
        )
        for r in results
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def build_answer(
    question: str,
    results: list[SearchResult],
    llm: LLMProvider,
    retrieval_hits: int = 0,
) -> AnswerResult:
    """
    Synchronous answer generation.

    Args:
        question:       The user's original question.
        results:        Re-ranked search results.
        llm:            LLM provider (injected).
        retrieval_hits: Total hits before re-ranking (for metadata).

    Returns:
        AnswerResult with full answer text and cited sources.
    """
    if not results:
        return AnswerResult(
            question=question,
            answer="I couldn't find relevant information in the knowledge base for this question. "
                   "Try rephrasing or check if the relevant documentation has been indexed.",
            sources=[],
            retrieval_hits=0,
        )

    messages = _build_prompt(question, results)
    answer = llm.complete(messages, temperature=0.1, max_tokens=2048)

    return AnswerResult(
        question=question,
        answer=answer,
        sources=_results_to_sources(results),
        retrieval_hits=retrieval_hits,
        metadata={"model": llm.model_name},
    )


def stream_answer(
    question: str,
    results: list[SearchResult],
    llm: LLMProvider,
    retrieval_hits: int = 0,
) -> Iterator[str]:
    """
    Streaming answer generation. Yields tokens then a JSON sentinel.

    The sentinel format (last yielded item):
        <!--RAG_META:{...}-->

    Consumers should strip the sentinel from the displayed text.
    """
    if not results:
        yield "I couldn't find relevant information in the knowledge base for this question."
        return

    messages = _build_prompt(question, results)

    for token in llm.stream(messages, temperature=0.1, max_tokens=2048):
        yield token

    # Yield metadata sentinel for the frontend
    sources_json = [
        {
            "title": s.title,
            "artifact_type": s.artifact_type,
            "doc_type": s.doc_type,
            "category": s.category,
            "repo": s.repo,
            "url": s.url,
            "score": round(s.score, 4),
            "image_refs": s.image_refs,
        }
        for s in _results_to_sources(results)
    ]
    sentinel = {
        "__sources__": sources_json,
        "__retrieval_hits__": retrieval_hits,
        "__model__": llm.model_name,
    }
    yield f"\n\n<!--RAG_META:{json.dumps(sentinel)}-->"
