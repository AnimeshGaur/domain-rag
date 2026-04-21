"""
processing/chunker.py
──────────────────────
Header-aware semantic chunker with Parent-Child support.

Enhancements over v2:
  • Phase 1: Split document into semantic sections at heading boundaries —
    chunks NEVER cross a heading boundary, so heading_path is always accurate.
  • Phase 2: Apply token-sized sub-splitting within each section, preserving
    code blocks as atomic units (never split mid-fence).
  • Short chunk filtering: chunks below MIN_CHUNK_TOKENS are discarded.
  • Stable content-addressed chunk IDs (sha → no re-indexing on re-run).
  • Image refs propagated from within each chunk.
  • Parent-Child Chunking: each section also emits a parent chunk (full
    section text) that child chunks reference via parent_id. At query time,
    the LLM receives the rich parent_text instead of the short child text.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import cfg
from processing.doc_schema import ProcessedDoc, Chunk

log = logging.getLogger(__name__)

# cl100k_base: used by all text-embedding-3 and GPT-4 family models
_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


# ── Regex patterns ────────────────────────────────────────────────────────────

# Split document at heading boundaries (H1–H4), keeping the heading with its section
_SECTION_SPLIT_RE = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)

# Detect fenced code blocks (``` ... ```)
_CODE_FENCE_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)

# For heading_path tracking
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Image links
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


# ── Sub-splitter (within a section) ──────────────────────────────────────────

_MD_SEPARATORS = [
    "\n\n",    # blank-line paragraph
    "\n",      # single newline
    ". ", "! ", "? ",  # sentence boundaries
    " ", "",   # word / char fallback
]

_sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.CHUNK_SIZE,
    chunk_overlap=cfg.CHUNK_OVERLAP,
    length_function=_token_len,
    separators=_MD_SEPARATORS,
    keep_separator=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}|{chunk_index}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _make_parent_id(doc_id: str, section_index: int) -> str:
    raw = f"{doc_id}|parent|{section_index}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _extract_heading_from_section(text: str) -> tuple[str, list[str]]:
    """
    Return (heading_text, remaining_lines) from the start of a section block.
    e.g. '## Auth\\nsome text' → ('Auth', ['some text'])
    """
    first_match = _HEADING_RE.match(text.lstrip())
    if first_match:
        return first_match.group(2).strip(), []
    return "", []


def _build_heading_path(heading_stack: list[tuple[int, str]]) -> str:
    """Build 'Title > Section > Subsection' breadcrumb from heading stack."""
    return " > ".join(h for _, h in heading_stack[:4])


def _image_refs_in(text: str) -> list[str]:
    """Return image paths referenced within a chunk."""
    return [m.group(1).strip() for m in _IMAGE_LINK_RE.finditer(text)]


def _prefix(doc: ProcessedDoc, heading_path: str) -> str:
    """
    Build the semantic prefix prepended to every chunk.
    Provides context for mid-document chunks in the embedding space.
    """
    parts = [
        f"[{doc.doc_type.upper()}]",
        f"Category: {doc.category}" if doc.category else "",
        f"Title: {doc.title}",
        f"Section: {heading_path}" if heading_path else "",
        "",  # blank line separator before body
    ]
    return "\n".join(p for p in parts if p != "" or p == parts[-1])


def _split_preserving_code_blocks(text: str, max_tokens: int) -> list[str]:
    """
    Split text while keeping fenced code blocks atomic.

    Strategy:
      1. Split text around code blocks (alternate: prose | code | prose ...).
      2. Sub-split prose parts with the standard token splitter.
      3. Emit code blocks as single chunks (even if oversized).
    """
    parts = _CODE_FENCE_RE.split(text)
    results: list[str] = []

    for part in parts:
        if not part.strip():
            continue
        if part.startswith("```"):
            # Code block — emit whole (never split)
            results.append(part.strip())
        else:
            if _token_len(part) <= max_tokens:
                if part.strip():
                    results.append(part.strip())
            else:
                sub_chunks = _sub_splitter.split_text(part)
                results.extend(c for c in sub_chunks if c.strip())

    return results


def _parse_heading_level(line: str) -> tuple[int, str] | None:
    """Parse a heading line and return (level, text) or None."""
    m = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


# ── Section-level splitter ────────────────────────────────────────────────────

def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split a document into (heading_path, section_text) pairs.

    Returns one tuple per heading-delimited section. The heading_path
    is the accurate breadcrumb for ALL content within that section.
    """
    raw_sections = _SECTION_SPLIT_RE.split(text)

    # Heading stack: [(level, heading_text), ...]
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # Determine if this section starts with a heading
        first_line = section.split("\n")[0]
        parsed = _parse_heading_level(first_line)

        if parsed:
            level, heading_text = parsed
            # Pop stale headings of same or lower priority
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))

        heading_path = _build_heading_path(heading_stack)
        sections.append((heading_path, section))

    # If no headings found, treat entire doc as one section
    if not sections:
        sections = [("", text)]

    return sections


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_doc(doc: ProcessedDoc, artifact_type: str = "") -> list[Chunk]:
    """
    Split a ProcessedDoc into Chunks using header-aware semantic splitting
    with Parent-Child architecture.

    Phase 1: Split at heading boundaries (accurate heading_path per chunk).
    Phase 2: Sub-split oversized sections, preserving code blocks atomically.
    Phase 3: Filter out chunks below MIN_CHUNK_TOKENS.
    Phase 4: Emit one "parent" chunk per section (full section text, not embedded)
             and link each child chunk to its parent via parent_id.

    Args:
        doc:           The processed document.
        artifact_type: Override for artifact_type field (set by LLM layer).
    """
    if not doc.clean_text.strip():
        log.debug("Skipping empty doc: %s", doc.path)
        return []

    sections = _split_into_sections(doc.clean_text)

    # Build child splits per section, tracking which section each belongs to
    section_child_splits: list[tuple[int, str, str]] = []  # (section_idx, heading_path, text)

    for sec_idx, (heading_path, section_text) in enumerate(sections):
        sub_chunks = _split_preserving_code_blocks(section_text, cfg.CHUNK_SIZE)
        for chunk_text in sub_chunks:
            section_child_splits.append((sec_idx, heading_path, chunk_text))

    # Filter short chunks
    section_child_splits = [
        (si, hp, ct) for si, hp, ct in section_child_splits
        if _token_len(ct) >= cfg.MIN_CHUNK_TOKENS
    ]

    total = len(section_child_splits)
    chunks: list[Chunk] = []
    child_chunk_idx = 0

    # Pre-compute parent IDs and section texts
    section_texts = {si: sec_text for si, (_, sec_text) in enumerate(sections)}

    for sec_idx, (heading_path, section_text) in enumerate(sections):
        # Build this section's parent chunk (full text, not sub-split)
        # Parent chunks are stored in ES but NOT embedded — they provide context
        parent_id = _make_parent_id(doc.doc_id, sec_idx)
        parent_prefix = _prefix(doc, heading_path)
        parent_full_text = f"{parent_prefix}\n{section_text}".strip()

        parent_chunk = Chunk(
            chunk_id=parent_id,
            doc_id=doc.doc_id,
            text=parent_full_text,
            title=doc.title,
            description=doc.description,
            chunk_index=-1,  # sentinel: not a searchable child
            total_chunks=total,
            heading_path=heading_path,
            doc_type=doc.doc_type,
            section=doc.section,
            category=doc.category,
            tags=doc.tags,
            artifact_type=artifact_type or doc.doc_type,
            repo=doc.repo,
            path=doc.path,
            url=doc.url,
            branch=doc.branch,
            image_refs=_image_refs_in(section_text),
            parent_id="",     # parents have no parent
            parent_text="",
            is_parent=True,
        )
        chunks.append(parent_chunk)

    # Now emit child chunks linked to their parent
    for sec_idx, heading_path, split in section_child_splits:
        parent_id = _make_parent_id(doc.doc_id, sec_idx)
        prefix = _prefix(doc, heading_path)
        enriched = f"{prefix}\n{split}".strip()
        img_refs = _image_refs_in(split)
        parent_full_text = section_texts.get(sec_idx, "")

        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.doc_id, child_chunk_idx),
            doc_id=doc.doc_id,
            text=enriched,
            title=doc.title,
            description=doc.description,
            chunk_index=child_chunk_idx,
            total_chunks=total,
            heading_path=heading_path,
            doc_type=doc.doc_type,
            section=doc.section,
            category=doc.category,
            tags=doc.tags,
            artifact_type=artifact_type or doc.doc_type,
            repo=doc.repo,
            path=doc.path,
            url=doc.url,
            branch=doc.branch,
            image_refs=img_refs,
            parent_id=parent_id,
            parent_text=parent_full_text,  # stored for LLM context expansion
            is_parent=False,
        ))
        child_chunk_idx += 1

    log.debug(
        "Doc '%s' → %d parent + %d child chunks",
        doc.title,
        len(sections),
        total,
    )
    return chunks


def chunk_docs(docs: list[ProcessedDoc]) -> list[Chunk]:
    """Chunk a list of ProcessedDocs and return a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_doc(doc))
    child_count = sum(1 for c in all_chunks if not c.is_parent)
    parent_count = sum(1 for c in all_chunks if c.is_parent)
    log.info(
        "Chunked %d docs → %d parent + %d child chunks (%d total)",
        len(docs), parent_count, child_count, len(all_chunks),
    )
    return all_chunks
