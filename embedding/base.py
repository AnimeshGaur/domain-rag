"""
embedding/base.py
──────────────────
Abstract EmbedProvider protocol.
Both OpenAI and local providers implement this interface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedProvider(Protocol):
    """
    Interface for embedding providers.
    Implement this to add new embedding backends (e.g. Cohere, Vertex AI).
    """

    @property
    def dims(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        May apply query-specific formatting (e.g. 'query: ' prefix for E5 models).

        Args:
            query: The search query.

        Returns:
            A single embedding vector.
        """
        ...


def get_embed_provider() -> EmbedProvider:
    """
    Factory function — returns the configured EmbedProvider.
    Import this instead of instantiating providers directly.

    Reads cfg.EMBED_PROVIDER:
      "openai"  → OpenAIEmbedProvider
      "local"   → LocalEmbedProvider (sentence-transformers)
    """
    from config import cfg

    if cfg.EMBED_PROVIDER == "openai":
        if not cfg.OPENAI_API_KEY:
            raise ValueError(
                "EMBED_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Either set the key or switch to EMBED_PROVIDER=local."
            )
        from embedding.openai_embed import OpenAIEmbedProvider
        return OpenAIEmbedProvider()

    if cfg.EMBED_PROVIDER == "local":
        from embedding.local_embed import LocalEmbedProvider
        return LocalEmbedProvider()

    raise ValueError(f"Unknown EMBED_PROVIDER: {cfg.EMBED_PROVIDER!r}. Use 'openai' or 'local'.")
