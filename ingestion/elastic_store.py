"""
ingestion/elastic_store.py
───────────────────────────
Manages the Elasticsearch index:
  • Creates / updates the mapping (dense_vector + BM25 text fields)
  • Upserts chunk documents with their embeddings
  • Exposes hybrid_search() used by the retrieval layer

Elasticsearch 8.x is required (uses native kNN + RRF).
"""
from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch, helpers

from config import cfg
from ingestion.chunker import Chunk

log = logging.getLogger(__name__)

es = Elasticsearch(cfg.ES_URL, request_timeout=60)

# ── Index mapping ─────────────────────────────────────────────────────────────

INDEX_MAPPING: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "rag_text": {
                    "type": "standard",
                    "stopwords": "_english_",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "chunk_id":      {"type": "keyword"},
            "artifact_type": {"type": "keyword"},
            "title":         {"type": "text", "analyzer": "rag_text",
                              "fields": {"keyword": {"type": "keyword"}}},
            "text":          {"type": "text", "analyzer": "rag_text"},
            "chunk_index":   {"type": "integer"},
            "total_chunks":  {"type": "integer"},
            "source_repo":   {"type": "keyword"},
            "source_paths":  {"type": "keyword"},
            "source_urls":   {"type": "keyword"},
            "language":      {"type": "keyword"},
            # Dense vector for kNN semantic search
            "embedding": {
                "type": "dense_vector",
                "dims": cfg.OPENAI_EMBED_DIMS,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def ensure_index() -> None:
    """Create index with mapping if it doesn't already exist."""
    if not es.indices.exists(index=cfg.ES_INDEX):
        es.indices.create(index=cfg.ES_INDEX, body=INDEX_MAPPING)
        log.info("Created Elasticsearch index: %s", cfg.ES_INDEX)
    else:
        log.info("Index already exists: %s", cfg.ES_INDEX)


def delete_index() -> None:
    """Drop the entire index (used for full re-ingestion)."""
    if es.indices.exists(index=cfg.ES_INDEX):
        es.indices.delete(index=cfg.ES_INDEX)
        log.info("Deleted index: %s", cfg.ES_INDEX)


def upsert_chunks(chunk_vector_pairs: list[tuple[Chunk, list[float]]]) -> int:
    """
    Bulk-upsert (chunk_id as document ID) chunks with their embeddings.
    Returns number of successfully indexed documents.
    """
    def _actions() -> Any:
        for chunk, vec in chunk_vector_pairs:
            doc = chunk.to_dict()
            doc["embedding"] = vec
            yield {
                "_op_type": "index",
                "_index": cfg.ES_INDEX,
                "_id": chunk.chunk_id,
                "_source": doc,
            }

    success, errors = helpers.bulk(
        es,
        _actions(),
        raise_on_error=False,
        stats_only=False,
    )
    if errors:
        log.warning("Bulk upsert had %d errors", len(errors))
    log.info("Indexed %d chunks into %s", success, cfg.ES_INDEX)
    return success


# ── Hybrid Search ─────────────────────────────────────────────────────────────

def hybrid_search(
    query_vec: list[float],
    query_text: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of:
      • kNN semantic search on the embedding field
      • BM25 full-text search on the text field

    Returns up to `top_k` results with _score and _source.
    """
    knn_clause = {
        "field": "embedding",
        "query_vector": query_vec,
        "k": top_k,
        "num_candidates": cfg.ES_KNN_CANDIDATES,
    }

    bm25_query: dict = {
        "bool": {
            "must": [{"match": {"text": {"query": query_text, "boost": 1.0}}}]
        }
    }

    # Apply optional metadata filters (e.g. {"artifact_type": "api_contract"})
    if filters:
        filter_clauses = [{"term": {k: v}} for k, v in filters.items()]
        bm25_query["bool"]["filter"] = filter_clauses
        knn_clause["filter"] = {"bool": {"must": filter_clauses}}

    body = {
        "size": top_k,
        "query": bm25_query,
        "knn": knn_clause,
        # RRF blends kNN + BM25 scores
        "rank": {
            "rrf": {
                "window_size": top_k * 2,
                "rank_constant": 60,
            }
        },
        "_source": {
            "excludes": ["embedding"]   # don't return the big vector
        },
    }

    resp = es.search(index=cfg.ES_INDEX, body=body)
    hits = resp["hits"]["hits"]
    return [{"score": h["_score"], **h["_source"]} for h in hits]


def index_stats() -> dict:
    """Return basic index stats for the /status endpoint."""
    try:
        stats = es.indices.stats(index=cfg.ES_INDEX)
        count = es.count(index=cfg.ES_INDEX)["count"]
        return {
            "index": cfg.ES_INDEX,
            "doc_count": count,
            "store_size_mb": round(
                stats["_all"]["primaries"]["store"]["size_in_bytes"] / 1e6, 2
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}
