"""
llm/openai_llm.py
──────────────────
OpenAI GPT-4o LLM provider.
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


class OpenAILLMProvider:
    """OpenAI chat completion provider (GPT-4o)."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI, RateLimitError, APIError
            self._OpenAI = OpenAI
            self._RateLimitError = RateLimitError
            self._APIError = APIError
        except ImportError:
            raise ImportError(
                "openai package is not installed. "
                "Run: pip install -r requirements-llm.txt"
            )
        self._client = OpenAI(api_key=cfg.OPENAI_API_KEY)

    @property
    def model_name(self) -> str:
        return cfg.OPENAI_CHAT_MODEL

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def complete(self, messages: list[dict], **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=cfg.OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=cfg.OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2048),
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
