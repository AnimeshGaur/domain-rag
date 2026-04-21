"""
llm/base.py
────────────
Abstract LLMProvider protocol + factory.
Import get_llm_provider() to get the configured provider.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM providers. Implement to add new backends."""

    @property
    def model_name(self) -> str:
        ...

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Synchronous completion. Returns the full response string."""
        ...

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """Streaming completion. Yields tokens one by one."""
        ...


def get_llm_provider() -> LLMProvider:
    """
    Factory function — returns the configured LLMProvider.
    Raises LLMNotEnabledError if LLM_PROVIDER=none.
    """
    from config import cfg

    if cfg.LLM_PROVIDER == "none" or not cfg.llm_enabled:
        raise LLMNotEnabledError(
            "LLM features are disabled (LLM_PROVIDER=none or OPENAI_API_KEY not set). "
            "To enable: set LLM_PROVIDER=openai and OPENAI_API_KEY in your .env file."
        )

    if cfg.LLM_PROVIDER == "openai":
        from llm.openai_llm import OpenAILLMProvider
        return OpenAILLMProvider()
        
    if cfg.LLM_PROVIDER == "groq":
        from llm.groq_llm import GroqLLMProvider
        return GroqLLMProvider()

    raise ValueError(f"Unknown LLM_PROVIDER: {cfg.LLM_PROVIDER!r}. Use 'openai', 'groq', or 'none'.")


class LLMNotEnabledError(RuntimeError):
    """Raised when an LLM operation is requested but no provider is configured."""
    pass
