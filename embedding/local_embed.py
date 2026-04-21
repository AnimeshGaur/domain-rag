"""
embedding/local_embed.py
─────────────────────────
Local sentence-transformers embedding provider.
Implements EmbedProvider protocol — zero API calls, fully offline.

Default model: BAAI/bge-small-en-v1.5 (384 dims, MTEB 62.2, much stronger on
               technical content than all-MiniLM-L6-v2)

Asymmetric query embedding (E5/BGE style):
  - Documents are embedded without prefix (chunk prefix in chunker is enough)
  - Queries are prefixed with BGE-recommended instruction string for better
    retrieval performance ("Represent this question for searching passages: ")

The model is lazy-loaded on first use and cached for the process lifetime.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from config import cfg

log = logging.getLogger(__name__)

# Known model → dims mapping (avoids loading model just for dims)
_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-mpnet-base-dot-v1": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
}

# BGE / E5 models use asymmetric query prefixes for better retrieval.
# Passage prefixes are already handled by the chunker's semantic prefix.
_QUERY_PREFIX: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this question for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5":  "Represent this question for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this question for searching relevant passages: ",
    "intfloat/e5-small-v2":   "query: ",
    "intfloat/e5-base-v2":    "query: ",
    "intfloat/e5-large-v2":   "query: ",
}


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Lazy-load and cache the SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    log.info("Loading local embedding model: %s (first load may download ~130MB)", model_name)
    model = SentenceTransformer(model_name)
    actual_dims = model.get_sentence_embedding_dimension()
    log.info("Model loaded. Embedding dims: %d", actual_dims)
    return model


class LocalEmbedProvider:
    """
    Sentence-Transformers local embedding provider with asymmetric query embedding.

    Supports BGE, E5, MiniLM, MPNet and other sentence-transformer models.
    Automatically applies model-specific query prefix for improved retrieval.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or cfg.LOCAL_EMBED_MODEL
        self._query_prefix = _QUERY_PREFIX.get(self._model_name, "")
        if self._query_prefix:
            log.debug("Using query prefix for %s: %r", self._model_name, self._query_prefix)

    @property
    def dims(self) -> int:
        # Use known dims map for fast startup; fall back to loading the model
        if self._model_name in _MODEL_DIMS:
            return _MODEL_DIMS[self._model_name]
        return _load_model(self._model_name).get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _model(self):
        return _load_model(self._model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of passage texts (documents / chunks).
        No query prefix applied — chunker prefix already provides context.
        """
        log.info("Local embed: %d texts with %s", len(texts), self._model_name)
        batch_size = cfg.EMBED_BATCH_SIZE
        vectors = self._model().encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # required for cosine similarity
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query with model-specific asymmetric prefix.
        BGE and E5 models are optimized for this asymmetric approach.
        """
        prefixed_query = f"{self._query_prefix}{query}" if self._query_prefix else query
        vec = self._model().encode(
            [prefixed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].tolist()
