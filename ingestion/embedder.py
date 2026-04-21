"""
ingestion/embedder.py
──────────────────────
Converts Chunk text → 3072-dim dense vectors using
OpenAI text-embedding-3-large.

Features:
  • Batches requests (max 100 texts per API call)
  • Retries on transient errors with exponential back-off
  • Returns (Chunk, vector) pairs ready for Elasticsearch
"""
from __future__ import annotations

import logging
import time
from typing import Iterator

from openai import OpenAI, RateLimitError, APIError

from config import cfg
from ingestion.chunker import Chunk

log = logging.getLogger(__name__)
client = OpenAI(api_key=cfg.OPENAI_API_KEY)

_BATCH_SIZE = 100       # OpenAI allows up to 2048 inputs; keep smaller for safety
_MAX_RETRIES = 5


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embedding API with retry."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(
                model=cfg.OPENAI_EMBED_MODEL,
                input=texts,
                dimensions=cfg.OPENAI_EMBED_DIMS,
            )
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        except RateLimitError:
            wait = 2 ** attempt
            log.warning("Rate limited on embed – retrying in %ds", wait)
            time.sleep(wait)
        except APIError as exc:
            log.error("OpenAI API error: %s", exc)
            raise
    raise RuntimeError("Exceeded max retries for embedding")


def embed_chunks(chunks: list[Chunk]) -> Iterator[tuple[Chunk, list[float]]]:
    """
    Yields (Chunk, embedding_vector) pairs.
    Processes in batches to stay within API limits.
    """
    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start: batch_start + _BATCH_SIZE]
        texts = [c.text for c in batch]

        log.info(
            "Embedding batch %d–%d / %d …",
            batch_start + 1,
            batch_start + len(batch),
            len(chunks),
        )
        vectors = _embed_batch(texts)

        for chunk, vec in zip(batch, vectors):
            yield chunk, vec


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    vecs = _embed_batch([query])
    return vecs[0]
