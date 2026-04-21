"""
llm/groq_llm.py
────────────────
Groq LLM provider using qwen/qwen3-32b or configured model.
Implements LLMProvider protocol.
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


class GroqLLMProvider:
    """Groq chat completion provider."""

    def __init__(self) -> None:
        try:
            from groq import Groq, RateLimitError, APIError
            self._Groq = Groq
            self._RateLimitError = RateLimitError
            self._APIError = APIError
        except ImportError:
            raise ImportError(
                "groq package is not installed. "
                "Run: pip install groq"
            )
        self._client = Groq(api_key=cfg.GROQ_API_KEY)

    @property
    def model_name(self) -> str:
        return cfg.GROQ_CHAT_MODEL

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def complete(self, messages: list[dict], **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=cfg.GROQ_CHAT_MODEL,
            messages=messages,
            temperature=kwargs.get("temperature", 0.6),
            max_completion_tokens=kwargs.get("max_tokens", 4096),
            top_p=0.95,
            stop=None,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=cfg.GROQ_CHAT_MODEL,
            messages=messages,
            temperature=kwargs.get("temperature", 0.6),
            max_completion_tokens=kwargs.get("max_tokens", 4096),
            top_p=0.95,
            stream=True,
            stop=None,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
