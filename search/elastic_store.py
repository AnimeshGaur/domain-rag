"""
search/elastic_store.py
────────────────────────
Manages the Elasticsearch index:
  • Creates / updates mapping (dense_vector + BM25 text fields, Docusaurus metadata)
  • Bulk-upserts chunks with their embeddings
  • Exposes index_stats() for monitoring

Requires Elasticsearch 8.x (uses native kNN + RRF rank).
"""
from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch, helpers

from config import cfg

log = logging.getLogger(__name__)

# Singleton ES client (lazy — created on first use)
_es: Elasticsearch | None = None


def _client() -> Elasticsearch:
    global _es
    if _es is None:
        _es = Elasticsearch(cfg.ES_URL, request_timeout=60)
    return _es


# ── Index mapping ─────────────────────────────────────────────────────────────

def _build_mapping(embed_dims: int) -> dict[str, Any]:
    """Build the index mapping for the given embedding dimensions."""
    return {
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
                # ── Content fields ─────────────────────────────
                "chunk_id":      {"type": "keyword"},
                "doc_id":        {"type": "keyword"},
                "title":         {
                    "type": "text",
                    "analyzer": "rag_text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "description":   {"type": "text", "analyzer": "rag_text"},
                "text":          {"type": "text", "analyzer": "rag_text"},
                "heading_path":  {
                    "type": "text",
                    "analyzer": "rag_text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                # ── Chunk position ─────────────────────────────
                "chunk_index":   {"type": "integer"},
                "total_chunks":  {"type": "integer"},
                # ── Parent-Child chunking ──────────────────────
                "parent_id":     {"type": "keyword"},   # points to parent chunk_id
                "is_parent":     {"type": "boolean"},   # True for parent chunks
                "parent_text":   {                      # full section text, stored not searched
                    "type": "text",
                    "index": False,
                },
                # ── Docusaurus metadata ────────────────────────
                "doc_type":      {"type": "keyword"},   # doc, blog, api_ref, concept, business…
                "section":       {"type": "keyword"},   # docs, blog, pages
                "category":      {"type": "keyword"},   # Docusaurus sidebar category
                "artifact_type": {"type": "keyword"},   # LLM artifact type (or doc_type)
                "tags":          {"type": "keyword"},   # multiple values
                # ── Provenance ─────────────────────────────────
                "repo":          {"type": "keyword"},
                "path":          {"type": "keyword"},
                "url":           {"type": "keyword"},
                "branch":        {"type": "keyword"},
                # ── Images ────────────────────────────────────
                "image_refs":    {"type": "keyword"},
                # ── Embedding vector ───────────────────────────
                "embedding": {
                    "type": "dense_vector",
                    "dims": embed_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }


# ── Index lifecycle ───────────────────────────────────────────────────────────

def ensure_index(embed_dims: int | None = None) -> None:
    """Create index with mapping if it doesn't already exist."""
    dims = embed_dims or cfg.embed_dims
    es = _client()
    if not es.indices.exists(index=cfg.ES_INDEX):
        es.indices.create(index=cfg.ES_INDEX, body=_build_mapping(dims))
        log.info("Created Elasticsearch index '%s' (dims=%d)", cfg.ES_INDEX, dims)
    else:
        log.info("Index '%s' already exists", cfg.ES_INDEX)


def delete_index() -> None:
    """Drop the index (used for full re-ingestion)."""
    es = _client()
    if es.indices.exists(index=cfg.ES_INDEX):
        es.indices.delete(index=cfg.ES_INDEX)
        log.info("Deleted index '%s'", cfg.ES_INDEX)


def recreate_index(embed_dims: int | None = None) -> None:
    """Delete + recreate the index (full reset)."""
    delete_index()
    ensure_index(embed_dims)


# ── Bulk indexing ─────────────────────────────────────────────────────────────

def upsert_chunks(
    chunk_vector_pairs: list[tuple[Any, list[float]]],
) -> int:
    """
    Bulk-upsert (chunk_id as document ID) chunks with their embeddings.
    Parent chunks (is_parent=True) are stored WITHOUT an embedding field —
    they exist only for context retrieval, not kNN search.
    Returns number of successfully indexed documents.
    """
    es = _client()

    def _actions():
        for chunk, vec in chunk_vector_pairs:
            doc = chunk.to_dict()
            if not chunk.is_parent:
                doc["embedding"] = vec  # only child chunks get a search vector
            # parent chunks: no embedding field — never participate in kNN
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
        log.warning("Bulk upsert had %d errors (first: %s)", len(errors), errors[0])
    log.info("Indexed %d chunks into '%s'", success, cfg.ES_INDEX)
    return int(success)


# ── Stats ─────────────────────────────────────────────────────────────────────

def index_stats() -> dict:
    """Return basic index stats for the /status endpoint."""
    try:
        es = _client()
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


def list_facets() -> dict:
    """
    Return facets for the /sources endpoint:
    distinct doc_types, categories, tags, and repos in the index.
    """
    try:
        es = _client()
        resp = es.search(
            index=cfg.ES_INDEX,
            body={
                "size": 0,
                "aggs": {
                    "doc_types":   {"terms": {"field": "doc_type",      "size": 20}},
                    "categories":  {"terms": {"field": "category",      "size": 50}},
                    "artifact_types": {"terms": {"field": "artifact_type", "size": 20}},
                    "tags":        {"terms": {"field": "tags",          "size": 50}},
                    "repos":       {"terms": {"field": "repo",          "size": 20}},
                },
            },
        )
        aggs = resp["aggregations"]
        return {
            "doc_types":      [b["key"] for b in aggs["doc_types"]["buckets"]],
            "categories":     [b["key"] for b in aggs["categories"]["buckets"]],
            "artifact_types": [b["key"] for b in aggs["artifact_types"]["buckets"]],
            "tags":           [b["key"] for b in aggs["tags"]["buckets"]],
            "repos":          [b["key"] for b in aggs["repos"]["buckets"]],
        }
    except Exception as exc:
        return {"error": str(exc)}
