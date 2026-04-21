"""
embedding/openai_embed.py
──────────────────────────
OpenAI text-embedding-3-large provider.
Implements EmbedProvider protocol.

Features:
  • Batched requests (100 texts per API call)
  • Exponential back-off via tenacity on rate-limit / transient errors
  • Configurable dimensions (1536 or 3072)
"""
from __future__ import annotations

import logging
from typing import Iterator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import cfg

log = logging.getLogger(__name__)

_BATCH_SIZE = 100   # OpenAI allows up to 2048; keep smaller for safety


class OpenAIEmbedProvider:
    """OpenAI text-embedding-3 provider."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI, RateLimitError, APIError
        except ImportError:
            raise ImportError(
                "openai package is not installed. "
                "Run: pip install -r requirements-llm.txt"
            )
        self._client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        self._RateLimitError = RateLimitError
        self._APIError = APIError

    @property
    def dims(self) -> int:
        return cfg.OPENAI_EMBED_DIMS

    @property
    def model_name(self) -> str:
        return cfg.OPENAI_EMBED_MODEL

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=cfg.OPENAI_EMBED_MODEL,
            input=texts,
            dimensions=cfg.OPENAI_EMBED_DIMS,
        )
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i: i + _BATCH_SIZE]
            log.info(
                "OpenAI embed batch %d–%d / %d",
                i + 1, i + len(batch), len(texts),
            )
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]
