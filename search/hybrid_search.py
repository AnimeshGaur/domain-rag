"""
search/hybrid_search.py
────────────────────────
Standalone hybrid search — NO LLM dependency.

Combines:
  • kNN semantic search on the dense embedding field
  • BM25 full-text search on text + title + heading_path
  • Manual Reciprocal Rank Fusion (RRF) merge in Python
    (ES Free tier doesn't support native RRF — we implement it ourselves)

Manual RRF formula (identical to ES platinum native RRF):
    score(d) = Σ  1 / (k + rank(d, source))
    where k=60 (standard constant)

Parent-Child awareness:
    Parent chunks (is_parent=True) are stored for LLM context expansion but
    are NEVER returned by kNN or BM25 — they are excluded via a filter clause.

Also provides:
  • search_by_keyword() — BM25-only for exact keyword queries
  • search_by_filter() — metadata-only filtered search
"""
from __future__ import annotations

import logging
from typing import Any

from config import cfg
from processing.doc_schema import SearchResult
from search.elastic_store import _client

log = logging.getLogger(__name__)

_RRF_K = 60  # Standard RRF constant

# Filter applied to all queries: never surface parent chunks in search results
_CHILD_ONLY = {"term": {"is_parent": False}}


# ── Manual RRF helper ─────────────────────────────────────────────────────────

def _manual_rrf_merge(
    knn_hits: list[dict],
    bm25_hits: list[dict],
    top_k: int,
    k: int = _RRF_K,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Works on any ES tier — no paid license required.

    RRF score = Σ  1 / (k + rank_in_source)
    """
    scores: dict[str, float] = {}
    hit_by_id: dict[str, dict] = {}

    for rank, hit in enumerate(knn_hits, start=1):
        doc_id = hit.get("_id", hit.get("_source", {}).get("chunk_id", ""))
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        hit_by_id[doc_id] = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit.get("_id", hit.get("_source", {}).get("chunk_id", ""))
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in hit_by_id:
            hit_by_id[doc_id] = hit

    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

    results = []
    for doc_id in sorted_ids[:top_k]:
        hit = dict(hit_by_id[doc_id])
        hit["_score"] = scores[doc_id]
        results.append(hit)

    return results


# ── Core hybrid search ────────────────────────────────────────────────────────

def hybrid_search(
    query_vec: list[float],
    query_text: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """
    Hybrid search: separate kNN + BM25 queries merged with manual Python RRF.
    Parent chunks are automatically excluded from all results.

    Args:
        query_vec:  Dense query embedding vector.
        query_text: Raw query text for BM25.
        top_k:      Maximum results to return.
        filters:    Optional metadata filters, e.g.
                    {"doc_type": "api_ref", "category": "Guides"}

    Returns:
        List of raw ES hit dicts (with _score=RRF score and _source).
    """
    es = _client()
    user_filters = _build_filter_clauses(filters)
    # Always exclude parent chunks from search — they're context only
    all_filters = [_CHILD_ONLY] + user_filters

    # kNN query
    knn_body: dict[str, Any] = {
        "size": top_k,
        "knn": {
            "field": "embedding",
            "query_vector": query_vec,
            "k": top_k,
            "num_candidates": cfg.ES_KNN_CANDIDATES,
            "filter": {"bool": {"must": all_filters}},
        },
        "_source": {"excludes": ["embedding"]},
    }

    # BM25 query
    bm25_body: dict[str, Any] = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "title^3",
                            "heading_path^2",
                            "description^1.5",
                            "text^1",
                            "tags^0.5",
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                }],
                "filter": all_filters,
            }
        },
        "_source": {"excludes": ["embedding"]},
    }

    knn_hits = es.search(index=cfg.ES_INDEX, body=knn_body)["hits"]["hits"]
    bm25_hits = es.search(index=cfg.ES_INDEX, body=bm25_body)["hits"]["hits"]

    merged = _manual_rrf_merge(knn_hits, bm25_hits, top_k=top_k)
    log.debug(
        "hybrid_search: kNN=%d, BM25=%d, RRF-merged=%d for query: %s",
        len(knn_hits), len(bm25_hits), len(merged), query_text[:80]
    )
    return merged


# ── BM25-only keyword search ──────────────────────────────────────────────────

def search_by_keyword(
    query_text: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """
    BM25 full-text search only — no embedding required.
    Useful for exact identifier lookups (API names, error codes, etc.).
    """
    es = _client()
    user_filters = _build_filter_clauses(filters)

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query_text,
                        "fields": ["title^3", "heading_path^2", "description^1.5", "text", "tags"],
                        "fuzziness": "AUTO",
                    }
                }],
                "filter": [_CHILD_ONLY] + user_filters,
            }
        },
        "_source": {"excludes": ["embedding"]},
    }

    resp = es.search(index=cfg.ES_INDEX, body=body)
    return resp["hits"]["hits"]


# ── Metadata-filter-only search ───────────────────────────────────────────────

def search_by_filter(
    filters: dict,
    top_k: int = 50,
) -> list[dict]:
    """
    Return documents matching metadata filters only (no text query).
    Useful for browsing all docs of a specific type or category.
    """
    es = _client()
    filter_clauses = [_CHILD_ONLY] + _build_filter_clauses(filters)

    body = {
        "size": top_k,
        "query": {"bool": {"filter": filter_clauses}},
        "sort": [{"chunk_index": "asc"}],
        "_source": {"excludes": ["embedding"]},
    }

    resp = es.search(index=cfg.ES_INDEX, body=body)
    return resp["hits"]["hits"]


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_filter_clauses(filters: dict | None) -> list[dict]:
    """Convert a filter dict into Elasticsearch filter clauses."""
    if not filters:
        return []
    clauses: list[dict] = []
    for key, value in filters.items():
        if isinstance(value, list):
            clauses.append({"terms": {key: value}})
        else:
            clauses.append({"term": {key: value}})
    return clauses


def hits_to_results(hits: list[dict]) -> list[SearchResult]:
    """Convert raw ES hits to SearchResult dataclasses."""
    return [SearchResult.from_hit(h) for h in hits]
