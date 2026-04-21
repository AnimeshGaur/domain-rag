"""
search/reranker.py
───────────────────
Cross-encoder re-ranking.
Standalone — no LLM dependency.

Uses sentence-transformers cross-encoder (ms-marco-MiniLM-L-6-v2 by default).
The model is lazy-loaded and cached for the process lifetime.

Given N candidates from hybrid search, returns the top-n re-ranked results.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from config import cfg

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load and cache the cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    log.info("Loading cross-encoder: %s", cfg.RERANKER_MODEL)
    return CrossEncoder(cfg.RERANKER_MODEL)


def rerank(
    query: str,
    hits: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """
    Re-rank ES hits using a cross-encoder model.

    Args:
        query:  The user's original query text.
        hits:   Raw ES hits (list of dicts with '_score' and '_source').
        top_n:  Number of top results to return. Defaults to cfg.RETRIEVAL_RERANK_TOP_N.

    Returns:
        Top-n hits sorted by cross-encoder score (descending),
        with 'rerank_score' added to each hit dict.
    """
    n = top_n or cfg.RETRIEVAL_RERANK_TOP_N

    if not hits:
        return []

    # Build (query, passage) pairs for cross-encoder
    passages = [h.get("_source", h).get("text", "") for h in hits]
    pairs = [(query, p[:1024]) for p in passages]  # truncate to avoid OOM

    try:
        model = _load_model()
        scores = model.predict(pairs)
    except Exception as exc:
        log.warning("Reranker failed (%s) — returning hits unranked", exc)
        return hits[:n]

    # Attach rerank score and sort descending
    scored = [
        {**h, "rerank_score": float(score)}
        for h, score in zip(hits, scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    log.debug(
        "Reranked %d → %d results (top score=%.4f)",
        len(hits), n, scored[0]["rerank_score"] if scored else 0,
    )
    return scored[:n]
