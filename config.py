"""
config.py
──────────
Central configuration — reads from environment / .env file.
Copy .env.example → .env and fill in your values.

Phase 1 (no LLM): only GITHUB_TOKEN + ES_URL needed.
Phase 2 (LLM):    also set OPENAI_API_KEY.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if it exists
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


def _list(key: str, default: str = "") -> list[str]:
    """Read a comma-separated env var into a list, stripping blanks."""
    return [v.strip() for v in os.environ.get(key, default).split(",") if v.strip()]


@dataclass
class Config:
    # ── Embedding ──────────────────────────────────────────────────────────────
    # "openai"  → text-embedding-3-large (requires OPENAI_API_KEY)
    # "local"   → sentence-transformers (fully offline, no API key needed)
    EMBED_PROVIDER: str = field(
        default_factory=lambda: os.environ.get("EMBED_PROVIDER", "openai")
    )
    # OpenAI embedding model + dimensions
    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
    OPENAI_EMBED_DIMS: int = field(
        default_factory=lambda: int(os.environ.get("OPENAI_EMBED_DIMS", "3072"))
    )
    # Local embedding model (used when EMBED_PROVIDER=local)
    LOCAL_EMBED_MODEL: str = field(
        default_factory=lambda: os.environ.get(
            "LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )
    LOCAL_EMBED_DIMS: int = 384  # bge-small-en-v1.5 output dims
    EMBED_BATCH_SIZE: int = field(
        default_factory=lambda: int(os.environ.get("EMBED_BATCH_SIZE", "64"))
    )

    # ── LLM (optional) ────────────────────────────────────────────────────────
    # "openai"  → GPT-4o  (requires OPENAI_API_KEY)
    # "groq"    → Qwen   (requires GROQ_API_KEY)
    # "none"    → LLM features disabled; /api/query returns 501
    LLM_PROVIDER: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "none")
    )
    OPENAI_CHAT_MODEL: str = field(
        default_factory=lambda: os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
    )
    GROQ_CHAT_MODEL: str = field(
        default_factory=lambda: os.environ.get("GROQ_CHAT_MODEL", "qwen/qwen3-32b")
    )

    # ── API credentials (shared by embed + LLM providers) ──────────────────
    OPENAI_API_KEY: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    GROQ_API_KEY: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )

    # ── GitHub ────────────────────────────────────────────────────────────────
    GITHUB_TOKEN: str = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN", "")
    )
    GITHUB_REPOS: list[str] = field(
        default_factory=lambda: _list("GITHUB_REPOS")
    )
    # Docusaurus docs folder inside the repo (e.g. "docs", "website/docs")
    DOCUSAURUS_DOCS_PATH: str = field(
        default_factory=lambda: os.environ.get("DOCUSAURUS_DOCS_PATH", "docs")
    )
    DOCUSAURUS_INCLUDE_BLOG: bool = field(
        default_factory=lambda: os.environ.get(
            "DOCUSAURUS_INCLUDE_BLOG", "false"
        ).lower() == "true"
    )
    GITHUB_INCLUDE_IMAGES: bool = field(
        default_factory=lambda: os.environ.get(
            "GITHUB_INCLUDE_IMAGES", "true"
        ).lower() == "true"
    )
    # Text extensions always crawled
    GITHUB_TEXT_EXTS: list[str] = field(
        default_factory=lambda: [
            ".md", ".mdx", ".txt", ".rst",
            ".yaml", ".yml", ".json", ".toml",
            ".js", ".jsx", ".ts", ".tsx",
            ".py", ".go", ".java",
        ]
    )
    # Image extensions (crawled when GITHUB_INCLUDE_IMAGES=true)
    GITHUB_IMAGE_EXTS: list[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"]
    )
    GITHUB_MAX_FILE_SIZE_KB: int = field(
        default_factory=lambda: int(
            os.environ.get("GITHUB_MAX_FILE_SIZE_KB", "500")
        )
    )

    # ── Elasticsearch ─────────────────────────────────────────────────────────
    ES_URL: str = field(
        default_factory=lambda: os.environ.get("ES_URL", "http://localhost:9200")
    )
    ES_INDEX: str = field(
        default_factory=lambda: os.environ.get("ES_INDEX", "ragbase_docs")
    )
    ES_KNN_CANDIDATES: int = field(
        default_factory=lambda: int(os.environ.get("ES_KNN_CANDIDATES", "100"))
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "400"))
    )
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "80"))
    )
    MIN_CHUNK_TOKENS: int = field(
        default_factory=lambda: int(os.environ.get("MIN_CHUNK_TOKENS", "20"))
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K_DENSE: int = 20
    RETRIEVAL_RERANK_TOP_N: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVAL_RERANK_TOP_N", "5"))
    )
    RERANKER_MODEL: str = field(
        default_factory=lambda: os.environ.get(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # ── Ingestion mode ────────────────────────────────────────────────────────
    # Override via CLI flag --no-llm / --llm or env var INGEST_USE_LLM
    INGEST_USE_LLM: bool = field(
        default_factory=lambda: os.environ.get(
            "INGEST_USE_LLM", "false"
        ).lower() == "true"
    )

    # ── LLM Artifact types ────────────────────────────────────────────────────
    ARTIFACT_TYPES: list[str] = field(
        default_factory=lambda: [
            "component_doc",
            "api_contract",
            "sequence_flow",
            "data_flow",
            "arch_summary",
            "business_process",
            "domain_model",
            "user_guide",
        ]
    )

    # ── Frontend ──────────────────────────────────────────────────────────────
    FRONTEND_ORIGIN: str = field(
        default_factory=lambda: os.environ.get(
            "FRONTEND_ORIGIN", "http://localhost:3000"
        )
    )

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def embed_dims(self) -> int:
        """Return the expected embedding dimensions for the active provider."""
        if self.EMBED_PROVIDER == "local":
            return self.LOCAL_EMBED_DIMS
        return self.OPENAI_EMBED_DIMS

    @property
    def llm_enabled(self) -> bool:
        if self.LLM_PROVIDER == "openai":
            return bool(self.OPENAI_API_KEY)
        if self.LLM_PROVIDER == "groq":
            return bool(self.GROQ_API_KEY)
        return False

    @property
    def openai_embed_enabled(self) -> bool:
        return self.EMBED_PROVIDER == "openai" and bool(self.OPENAI_API_KEY)


cfg = Config()
