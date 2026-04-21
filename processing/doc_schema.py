"""
processing/doc_schema.py
─────────────────────────
Shared dataclasses used across crawling, processing, embedding, and search layers.
No external API dependencies — pure Python.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


# ── Raw content from GitHub ───────────────────────────────────────────────────

@dataclass
class ImageRef:
    """An image file or an image referenced inside a markdown file."""
    path: str               # file path in repo (e.g. "docs/img/arch.png")
    url: str                # GitHub raw/blob URL
    alt_text: str = ""      # from markdown ![alt](...) syntax
    caption: str = ""       # from Docusaurus figure captions or surrounding text
    surrounding_text: str = ""  # the paragraph that contains this image link


@dataclass
class RawDoc:
    """
    A single crawled document from GitHub.
    Produced by crawling/ layer. Consumed by processing/ layer.
    """
    repo: str               # "owner/repo"
    path: str               # file path inside repo
    content: str            # raw UTF-8 text (markdown / MDX / code)
    sha: str                # git blob SHA (for change detection)
    url: str                # GitHub HTML URL (for citation links)
    branch: str             # default branch name
    # Docusaurus-specific metadata
    doc_type: str = "doc"   # "doc" | "blog" | "page" | "image" | "config"
    section: str = ""       # "docs" | "blog" | "src/pages" (top-level folder)
    category: str = ""      # Docusaurus sidebar category label
    frontmatter: dict = field(default_factory=dict)   # parsed YAML frontmatter
    images: list[ImageRef] = field(default_factory=list)  # images linked in this doc


# ── Processed / ready-to-chunk ────────────────────────────────────────────────

@dataclass
class ProcessedDoc:
    """
    Cleaned document ready for chunking.
    Produced by processing/markdown_processor. Consumed by processing/chunker.
    """
    # provenance (carried through from RawDoc)
    repo: str
    path: str
    url: str
    sha: str
    branch: str
    doc_type: str
    section: str
    category: str
    frontmatter: dict
    images: list[ImageRef]

    # processed content
    title: str              # resolved title (frontmatter > first H1 > filename)
    description: str        # frontmatter description or first paragraph
    clean_text: str         # MDX-stripped, normalized markdown
    heading_tree: list[str] # ["H1 text", "H2 text", ...] — heading hierarchy
    tags: list[str]         # frontmatter tags + inferred tags
    links: list[str]        # internal doc links (for link graph)

    @property
    def heading_path(self) -> str:
        """Breadcrumb string like 'Guide > Authentication > OAuth2'."""
        return " > ".join(self.heading_tree)

    @property
    def doc_id(self) -> str:
        """Stable content-addressed ID based on repo + path."""
        return hashlib.sha256(f"{self.repo}|{self.path}".encode()).hexdigest()[:16]


# ── Chunk (indexable unit) ────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single chunk ready for embedding and indexing into Elasticsearch.
    Produced by processing/chunker. Consumed by embedding/ and search/ layers.
    """
    chunk_id: str           # SHA-256(doc_id + chunk_index)[:24]
    doc_id: str             # parent document ID
    # content
    text: str               # chunk text (prefixed with heading context)
    title: str              # document title
    description: str        # document description
    # position metadata
    chunk_index: int
    total_chunks: int
    heading_path: str       # "Guide > Auth > OAuth2" at the point of this chunk
    # document metadata
    doc_type: str           # "doc" | "blog" | "page"
    section: str            # "docs" | "blog"
    category: str           # Docusaurus sidebar category
    tags: list[str]
    # provenance
    repo: str
    path: str
    url: str
    branch: str
    # artifact type (set by LLM layer; defaults to doc_type)
    artifact_type: str = ""
    # image references contained in or near this chunk
    image_refs: list[str] = field(default_factory=list)
    # Parent-Child chunking fields
    parent_id: str = ""        # chunk_id of the parent section chunk (empty for parents)
    parent_text: str = ""      # full section text for LLM context (stored, not embedded)
    is_parent: bool = False    # True for parent chunks — skipped during embedding/kNN

    def to_dict(self) -> dict:
        return {
            "chunk_id":     self.chunk_id,
            "doc_id":       self.doc_id,
            "text":         self.text,
            "title":        self.title,
            "description":  self.description,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
            "heading_path": self.heading_path,
            "doc_type":     self.doc_type,
            "section":      self.section,
            "category":     self.category,
            "tags":         self.tags,
            "artifact_type": self.artifact_type or self.doc_type,
            "repo":         self.repo,
            "path":         self.path,
            "url":          self.url,
            "branch":       self.branch,
            "image_refs":   self.image_refs,
            "parent_id":    self.parent_id,
            "parent_text":  self.parent_text,
            "is_parent":    self.is_parent,
        }


# ── Search result ─────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search hit returned by the search layer."""
    chunk_id: str
    doc_id: str
    title: str
    description: str
    text: str
    score: float
    rerank_score: Optional[float]
    doc_type: str
    section: str
    category: str
    artifact_type: str
    tags: list[str]
    repo: str
    path: str
    url: str
    heading_path: str
    image_refs: list[str]
    chunk_index: int
    total_chunks: int
    # Parent-Child context fields
    parent_id: str = ""
    parent_text: str = ""
    is_parent: bool = False

    @classmethod
    def from_hit(cls, hit: dict, rerank_score: float | None = None) -> "SearchResult":
        src = hit.get("_source", hit)
        return cls(
            chunk_id=src.get("chunk_id", ""),
            doc_id=src.get("doc_id", ""),
            title=src.get("title", ""),
            description=src.get("description", ""),
            text=src.get("text", ""),
            score=hit.get("_score", src.get("score", 0.0)),
            rerank_score=rerank_score,
            doc_type=src.get("doc_type", ""),
            section=src.get("section", ""),
            category=src.get("category", ""),
            artifact_type=src.get("artifact_type", ""),
            tags=src.get("tags", []),
            repo=src.get("repo", ""),
            path=src.get("path", ""),
            url=src.get("url", ""),
            heading_path=src.get("heading_path", ""),
            image_refs=src.get("image_refs", []),
            chunk_index=src.get("chunk_index", 0),
            total_chunks=src.get("total_chunks", 1),
            parent_id=src.get("parent_id", ""),
            parent_text=src.get("parent_text", ""),
            is_parent=src.get("is_parent", False),
        )

