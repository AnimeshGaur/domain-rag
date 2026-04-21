"""
ingestion/chunker.py
─────────────────────
Splits Artifact documents into overlapping token-aware chunks.

Strategy:
  1. Prefer splitting on Markdown headings → paragraphs → sentences → words
  2. Target ~512 tokens with ~10 % overlap
  3. Each chunk retains full provenance metadata for citation

Uses LangChain's RecursiveCharacterTextSplitter with tiktoken for
accurate GPT-family token counting.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import cfg
from ingestion.artifact_generator import Artifact

log = logging.getLogger(__name__)

# GPT-4 / embedding-3 both use cl100k_base
_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Chunk:
    chunk_id: str           # SHA-256 of (artifact_id + chunk_index)
    artifact_type: str
    title: str              # artifact title (for display)
    text: str               # the actual chunk text
    chunk_index: int        # position within the artifact
    total_chunks: int       # total chunks from this artifact (filled later)
    source_repo: str
    source_paths: list[str]
    source_urls: list[str]
    language: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id":      self.chunk_id,
            "artifact_type": self.artifact_type,
            "title":         self.title,
            "text":          self.text,
            "chunk_index":   self.chunk_index,
            "total_chunks":  self.total_chunks,
            "source_repo":   self.source_repo,
            "source_paths":  self.source_paths,
            "source_urls":   self.source_urls,
            "language":      self.language,
            **self.metadata,
        }


# ── Splitter setup ────────────────────────────────────────────────────────────

# Markdown-aware separators: headings first, then paragraphs, then lines
_MD_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ",   # heading boundaries
    "\n\n",                          # blank-line paragraphs
    "\n",                            # single newlines
    ". ", "! ", "? ",                # sentence boundaries
    " ", "",                         # word / char fallback
]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.CHUNK_SIZE,
    chunk_overlap=cfg.CHUNK_OVERLAP,
    length_function=_token_len,
    separators=_MD_SEPARATORS,
    keep_separator=True,
)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_artifact(artifact: Artifact) -> list[Chunk]:
    """
    Split one Artifact into a list of Chunks.
    The artifact title is prepended to every chunk so the embedding model
    always sees the semantic context of the section.
    """
    raw_splits = _splitter.split_text(artifact.content)
    total = len(raw_splits)
    chunks: list[Chunk] = []

    for idx, text in enumerate(raw_splits):
        # Prepend the artifact title for richer embeddings
        enriched = f"[{artifact.artifact_type.upper()}] {artifact.title}\n\n{text}"

        # Stable ID: hash of repo + title + index
        id_src = f"{artifact.source_repo}|{artifact.title}|{idx}".encode()
        chunk_id = hashlib.sha256(id_src).hexdigest()[:24]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            artifact_type=artifact.artifact_type,
            title=artifact.title,
            text=enriched,
            chunk_index=idx,
            total_chunks=total,
            source_repo=artifact.source_repo,
            source_paths=artifact.source_paths,
            source_urls=artifact.source_urls,
            language=artifact.language,
            metadata=artifact.metadata,
        ))

    log.debug(
        "Artifact '%s' → %d chunks (avg %.0f tokens)",
        artifact.title,
        total,
        sum(_token_len(c.text) for c in chunks) / max(total, 1),
    )
    return chunks


def chunk_artifacts(artifacts: list[Artifact]) -> list[Chunk]:
    """Chunk all artifacts and return a flat list."""
    all_chunks: list[Chunk] = []
    for art in artifacts:
        all_chunks.extend(chunk_artifact(art))
    log.info(
        "Chunked %d artifacts → %d total chunks",
        len(artifacts), len(all_chunks),
    )
    return all_chunks
