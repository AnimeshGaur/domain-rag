"""
search/query_expansion.py
──────────────────────────
Multi-Query Expansion for improved retrieval recall.

Instead of running a single hybrid search for the user's raw question,
this module:
  1. Asks the LLM to rephrase the question into N semantically distinct variants
  2. Runs a full hybrid search for EACH variant in sequence
  3. Merges all result lists using a final RRF pass

This dramatically improves recall for:
  - Ambiguous queries ("doctor module" → "physician scheduling", "medical staff")
  - Domain-specific jargon where users and docs use different terminology
  - Queries where one phrasing favors kNN and another favors BM25

Phase 1 compatibility:
    If llm is None, returns the original question as the only variant.
    All callers degrade gracefully to single-query search.

Reference: RAG-Fusion (Shi et al., 2023) — multi-query + RRF fusion.
"""
from __future__ import annotations

import hashlib
import logging
import time

log = logging.getLogger(__name__)

_EXPANSION_CACHE: dict[str, tuple[list[str], float]] = {}
_CACHE_TTL_SECONDS = 300

_EXPANSION_SYSTEM = (
    "You are a query rewriter for a technical documentation search engine. "
    "Given a user's question, produce exactly {n} alternative phrasings that capture "
    "different angles of the same information need. "
    "Rules:\n"
    "  - Each rephrasing must be semantically distinct (different words/framing)\n"
    "  - Keep them concise (under 20 words each)\n"
    "  - Vary the vocabulary: technical terms, plain English, action-oriented\n"
    "  - Output ONLY the questions, one per line, no numbering or bullets\n"
    "  - Do NOT include the original question"
)


def _cache_key(question: str, n: int, model: str) -> str:
    return hashlib.md5(f"{model}::{n}::{question}".encode()).hexdigest()


def expand_query(question: str, llm, n: int = 3) -> list[str]:
    """
    Generate N semantically distinct rephrasings of the input question.

    Results are cached per (question, n, model) for _CACHE_TTL_SECONDS.

    Args:
        question: Raw user query.
        llm:      LLMProvider instance.
        n:        Number of variants to generate (default: 3).

    Returns:
        List of query variants including the original question.
        (original question is always included as the first element)
    """
    if llm is None:
        return [question]

    key = _cache_key(question, n, llm.model_name)
    now = time.monotonic()

    if key in _EXPANSION_CACHE:
        cached_variants, cached_at = _EXPANSION_CACHE[key]
        if now - cached_at < _CACHE_TTL_SECONDS:
            log.debug("Query expansion cache hit for %r", question[:60])
            return [question] + cached_variants

    messages = [
        {"role": "system", "content": _EXPANSION_SYSTEM.format(n=n)},
        {"role": "user", "content": question},
    ]

    try:
        raw = llm.complete(messages, temperature=0.7, max_tokens=150)
        variants = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and line.strip() != question
        ][:n]
        log.debug("Query expansion: %d variants for %r", len(variants), question[:60])
    except Exception as exc:
        log.warning("Query expansion failed (%s) — using original query only", exc)
        return [question]

    # Cache only the variants (original is always prepended at read time)
    _EXPANSION_CACHE[key] = (variants, now)
    if len(_EXPANSION_CACHE) > 500:
        cutoff = now - _CACHE_TTL_SECONDS
        stale = [k for k, (_, t) in _EXPANSION_CACHE.items() if t < cutoff]
        for k in stale:
            del _EXPANSION_CACHE[k]

    return [question] + variants


def multi_query_search(
    variants: list[str],
    embed_provider,
    filters: dict | None,
    top_k: int,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Run a full hybrid search for each query variant, then merge all
    result lists using Reciprocal Rank Fusion.

    This is the same RRF formula used in hybrid_search.py — applied
    at the cross-query level: score(d) = Σ 1 / (rrf_k + rank(d, variant_i))

    Args:
        variants:       List of query strings (original + rephrased).
        embed_provider: EmbedProvider for vectorising each variant.
        filters:        Metadata filters passed to hybrid_search.
        top_k:          Candidates per variant (not total).
        rrf_k:          RRF smoothing constant (default: 60).

    Returns:
        Merged, deduplicated list of ES hits sorted by final RRF score.
    """
    from search.hybrid_search import hybrid_search

    all_hit_lists: list[list[dict]] = []

    for variant in variants:
        try:
            q_vec = embed_provider.embed_query(variant)
            hits = hybrid_search(
                query_vec=q_vec,
                query_text=variant,
                top_k=top_k,
                filters=filters,
            )
            all_hit_lists.append(hits)
            log.debug("multi_query_search: %d hits for variant %r", len(hits), variant[:60])
        except Exception as exc:
            log.warning("Variant search failed for %r: %s", variant[:60], exc)

    if not all_hit_lists:
        return []

    # Final cross-query RRF merge
    scores: dict[str, float] = {}
    hit_by_id: dict[str, dict] = {}

    for hit_list in all_hit_lists:
        for rank, hit in enumerate(hit_list, start=1):
            doc_id = hit.get("_id", hit.get("_source", {}).get("chunk_id", ""))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in hit_by_id:
                hit_by_id[doc_id] = hit

    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    merged = []
    for doc_id in sorted_ids[:top_k]:
        hit = dict(hit_by_id[doc_id])
        hit["_score"] = scores[doc_id]
        merged.append(hit)

    log.info(
        "multi_query_search: %d variants × ~%d hits → %d unique after RRF",
        len(variants), top_k, len(merged),
    )
    return merged
