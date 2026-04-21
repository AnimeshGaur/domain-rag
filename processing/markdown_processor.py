"""
processing/markdown_processor.py
──────────────────────────────────
Transforms RawDoc objects → ProcessedDoc objects:
  • Strips MDX-specific syntax (JSX imports, component tags)
  • Normalizes heading hierarchy and builds the heading tree
  • Resolves title, description, tags
  • Extracts internal links for a link graph
  • Produces clean, chunk-ready markdown text

No external API calls. Pure Python.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from processing.doc_schema import RawDoc, ProcessedDoc
from crawling.docusaurus_parser import (
    parse_frontmatter,
    extract_title,
    extract_description,
    extract_tags,
    extract_image_refs,
)

log = logging.getLogger(__name__)

# ── MDX / JSX stripping regexes ───────────────────────────────────────────────

# Remove MDX import statements (e.g. "import Tabs from '@theme/Tabs';")
_MDX_IMPORT_RE = re.compile(r"^import\s+\S+.*$", re.MULTILINE)

# Remove JSX component tags (self-closing and block)
_JSX_SELF_CLOSE_RE = re.compile(r"<[A-Z][a-zA-Z]*[^>]*/\s*>", re.DOTALL)
_JSX_OPEN_CLOSE_RE = re.compile(
    r"<([A-Z][a-zA-Z]*)(?:[^>]*)>(.*?)</\1>",
    re.DOTALL,
)
# Keep the inner text of JSX containers (e.g. <Tabs>, <TabItem>)
# But strip the tags themselves

# Remove HTML style/script blocks
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)

# Admonition syntax (:::note, :::tip, etc.) — keep content, strip markers
_ADMONITION_OPEN_RE = re.compile(r"^:::(\w+)(?:\s+.*)?$", re.MULTILINE)
_ADMONITION_CLOSE_RE = re.compile(r"^:::$", re.MULTILINE)

# MDX export statements
_MDX_EXPORT_RE = re.compile(r"^export\s+.*$", re.MULTILINE)

# Heading extraction
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Internal Docusaurus links ([text](./relative-path))
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?!http)([^)]+)\)", re.MULTILINE)


# ── Processing functions ──────────────────────────────────────────────────────

def strip_mdx(text: str) -> str:
    """Remove MDX/JSX syntax while preserving prose and code content."""
    # Remove imports and exports
    text = _MDX_IMPORT_RE.sub("", text)
    text = _MDX_EXPORT_RE.sub("", text)
    # Remove script/style blocks
    text = _SCRIPT_STYLE_RE.sub("", text)
    # Unwrap JSX containers (keep inner text)
    text = _JSX_OPEN_CLOSE_RE.sub(r"\2", text)
    # Remove remaining self-closing JSX tags
    text = _JSX_SELF_CLOSE_RE.sub("", text)
    # Clean admonition syntax (keep body)
    text = _ADMONITION_OPEN_RE.sub("", text)
    text = _ADMONITION_CLOSE_RE.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_heading_tree(text: str) -> list[str]:
    """
    Extract the heading hierarchy as a flat list: ["H1 title", "H2 subtitle", ...].
    Used to build heading_path for each chunk.
    """
    headings: list[str] = []
    for match in _HEADING_RE.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append(title)
    return headings


def extract_internal_links(text: str, base_path: str) -> list[str]:
    """Extract relative links from markdown (for future link graph analysis)."""
    links: list[str] = []
    for match in _LINK_RE.finditer(text):
        href = match.group(2).strip()
        # Normalize
        if not href.startswith("#"):
            links.append(href)
    return list(set(links))


def infer_tags_from_content(text: str, existing_tags: list[str]) -> list[str]:
    """
    Infer additional tags from content signals.
    Useful for docs that have no explicit frontmatter tags.
    """
    extra: list[str] = []
    lower = text.lower()
    signals = {
        "authentication": ["auth", "oauth", "jwt", "session", "token"],
        "api": ["endpoint", "rest", "graphql", "api key", "request", "response"],
        "database": ["sql", "database", "schema", "migration", "query"],
        "architecture": ["architecture", "design pattern", "microservice", "monolith"],
        "deployment": ["docker", "kubernetes", "ci/cd", "deploy", "pipeline"],
        "testing": ["unit test", "integration test", "pytest", "jest"],
        "business-process": ["workflow", "business process", "approval", "stakeholder"],
    }
    for tag, keywords in signals.items():
        if tag not in existing_tags and any(kw in lower for kw in keywords):
            extra.append(tag)
    return existing_tags + extra


# ── Public API ────────────────────────────────────────────────────────────────

def process_doc(raw: RawDoc) -> ProcessedDoc:
    """
    Transform a RawDoc into a ProcessedDoc ready for chunking.

    For markdown/MDX files: parses frontmatter, strips JSX, extracts structure.
    For other file types (code, config): minimal processing.
    """
    path = raw.path
    ext = Path(path).suffix.lower()
    is_markdown = ext in (".md", ".mdx")

    if is_markdown:
        frontmatter, body = parse_frontmatter(raw.content)
        clean_text = strip_mdx(body)
    else:
        frontmatter = raw.frontmatter
        clean_text = raw.content

    # Merge frontmatter from raw.frontmatter (already parsed by crawler) if available
    if raw.frontmatter:
        frontmatter = {**raw.frontmatter, **frontmatter}

    title = extract_title(frontmatter, clean_text, path)
    description = extract_description(frontmatter, clean_text)
    tags = extract_tags(frontmatter)
    tags = infer_tags_from_content(clean_text, tags)
    heading_tree = extract_heading_tree(clean_text)
    links = extract_internal_links(clean_text, path)

    return ProcessedDoc(
        repo=raw.repo,
        path=raw.path,
        url=raw.url,
        sha=raw.sha,
        branch=raw.branch,
        doc_type=raw.doc_type,
        section=raw.section,
        category=raw.category,
        frontmatter=frontmatter,
        images=raw.images,
        title=title,
        description=description,
        clean_text=clean_text,
        heading_tree=heading_tree,
        tags=tags,
        links=links,
    )


def process_docs(raw_docs: list[RawDoc]) -> list[ProcessedDoc]:
    """Process a list of RawDocs in batch."""
    processed: list[ProcessedDoc] = []
    for raw in raw_docs:
        try:
            processed.append(process_doc(raw))
        except Exception as exc:
            log.warning("Failed to process %s: %s", raw.path, exc)
    log.info("Processed %d / %d docs", len(processed), len(raw_docs))
    return processed
