"""
retrieval/reranker.py
──────────────────────
Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2).

Takes the top-K results from hybrid search and re-scores them by
feeding (query, chunk_text) pairs through a bi-directional
cross-attention model.  Much more accurate than vector cosine alone.

Lazy-loads the model on first use to avoid slowing startup.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config import cfg

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model():
    """Load cross-encoder model once and cache in memory."""
    try:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder: %s", cfg.RERANKER_MODEL)
        return CrossEncoder(cfg.RERANKER_MODEL, max_length=512)
    except ImportError:
        log.warning(
            "sentence-transformers not installed – re-ranking disabled. "
            "pip install sentence-transformers"
        )
        return None


def rerank(query: str, hits: list[dict], top_n: int | None = None) -> list[dict]:
    """
    Re-score `hits` (each must have a "text" field) using the cross-encoder.
    Returns `top_n` results sorted by descending cross-encoder score.

    Falls back to original order if the model isn't available.
    """
    n = top_n or cfg.RETRIEVAL_RERANK_TOP_N
    model = _get_model()

    if model is None or not hits:
        return hits[:n]

    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    log.debug(
        "Re-ranked %d hits → top %d  (top score: %.3f)",
        len(hits), n, ranked[0]["rerank_score"] if ranked else 0,
    )
    return ranked[:n]
