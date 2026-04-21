"""
search/hyde.py
───────────────
HyDE — Hypothetical Document Embeddings

Instead of embedding the raw user question (short, question-shaped),
we ask the LLM to write a plausible short passage that ANSWERS the question.
That passage is structurally similar to real indexed documents, so kNN
recall is significantly improved.

Reference: Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without
           Relevance Labels" (https://arxiv.org/abs/2212.10496)

Usage:
    from search.hyde import hyde_embed
    q_vec = hyde_embed("How does the auth flow work?", llm, embed_provider)

Phase 1 compatibility:
    If llm is None, falls back to standard embed_query() — no regression.
"""
from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# Simple in-memory LRU cache keyed on (question, model_name)
# Prevents redundant LLM calls for the same query within a session
_HYPOTHESIS_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


_HYDE_SYSTEM = (
    "You are a technical documentation assistant. "
    "Write a single concise paragraph (3-5 sentences) that directly answers "
    "the user's question as if you were a relevant section of documentation. "
    "Use the same vocabulary and style as technical docs. "
    "Do NOT explain that you're generating a hypothesis — just write the passage."
)


def _cache_key(question: str, model: str) -> str:
    return hashlib.md5(f"{model}::{question}".encode()).hexdigest()


def generate_hypothesis(question: str, llm) -> str:
    """
    Use the LLM to generate a short hypothetical passage that answers
    the question. The passage is used as the embedding subject instead
    of the raw query.

    Results are cached for _CACHE_TTL_SECONDS to avoid redundant LLM calls.

    Args:
        question: The user's raw query string.
        llm:      Any LLMProvider (Groq, OpenAI, etc.)

    Returns:
        A short passage string suitable for embedding.
    """
    key = _cache_key(question, llm.model_name)
    now = time.monotonic()

    # Check cache
    if key in _HYPOTHESIS_CACHE:
        cached_text, cached_at = _HYPOTHESIS_CACHE[key]
        if now - cached_at < _CACHE_TTL_SECONDS:
            log.debug("HyDE cache hit for %r", question[:60])
            return cached_text

    messages = [
        {"role": "system", "content": _HYDE_SYSTEM},
        {"role": "user", "content": question},
    ]

    try:
        hypothesis = llm.complete(messages, temperature=0.5, max_tokens=200)
        hypothesis = hypothesis.strip()
        log.debug("HyDE hypothesis (%d chars): %s…", len(hypothesis), hypothesis[:80])
    except Exception as exc:
        log.warning("HyDE generation failed (%s) — falling back to raw query", exc)
        return question

    # Store in cache
    _HYPOTHESIS_CACHE[key] = (hypothesis, now)
    # Evict stale entries if cache growing large
    if len(_HYPOTHESIS_CACHE) > 500:
        cutoff = now - _CACHE_TTL_SECONDS
        stale = [k for k, (_, t) in _HYPOTHESIS_CACHE.items() if t < cutoff]
        for k in stale:
            del _HYPOTHESIS_CACHE[k]

    return hypothesis


def hyde_embed(
    question: str,
    llm,
    embed_provider,
) -> list[float]:
    """
    Full HyDE pipeline:
      1. Generate a hypothetical answer passage via LLM
      2. Embed it using the same embed_provider as the index

    Falls back to standard embed_query if LLM raises or is None.

    Args:
        question:      Raw user query.
        llm:           LLMProvider instance (or None for Phase 1 fallback).
        embed_provider: EmbedProvider (LocalEmbedProvider or OpenAIEmbedProvider).

    Returns:
        Dense vector (list[float]) for kNN search.
    """
    if llm is None:
        return embed_provider.embed_query(question)

    hypothesis = generate_hypothesis(question, llm)

    # Use embed_query (applies BGE asymmetric prefix automatically)
    return embed_provider.embed_query(hypothesis)
