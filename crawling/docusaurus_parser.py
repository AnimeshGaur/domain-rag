"""
crawling/docusaurus_parser.py
──────────────────────────────
Parses Docusaurus-specific metadata from crawled files:
  • YAML / TOML frontmatter (title, description, tags, sidebar_label)
  • sidebar.js / sidebars.json → category hierarchy
  • doc_type detection from folder path
  • image reference extraction from markdown content
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import frontmatter as fm

from processing.doc_schema import ImageRef

log = logging.getLogger(__name__)

# ── Doc-type detection ────────────────────────────────────────────────────────

_SECTION_MAP: dict[str, str] = {
    "docs":        "docs",
    "blog":        "blog",
    "src/pages":   "pages",
    "pages":       "pages",
    "static":      "static",
}


def detect_section(path: str, docs_path: str = "docs") -> str:
    """Returns the top-level section: 'docs', 'blog', 'pages', 'static', or 'other'."""
    p = path.lower().replace("\\", "/")
    for prefix, section in _SECTION_MAP.items():
        if p.startswith(prefix + "/") or p == prefix:
            return section
    if p.startswith(docs_path.lower()):
        return "docs"
    return "other"


def detect_doc_type(path: str, frontmatter: dict) -> str:
    """
    Infer doc_type from path + frontmatter:
      doc, tutorial, guide, api_ref, concept, business, blog, page, config
    """
    p = path.lower()
    fm_tags = " ".join(str(t) for t in frontmatter.get("tags", []) or []).lower()
    fm_cat = str(frontmatter.get("sidebar_label", "")).lower()

    if "/blog/" in p or p.startswith("blog/"):
        return "blog"
    if "/src/pages/" in p or p.startswith("src/pages/"):
        return "page"

    # Detect from frontmatter tags
    if any(t in fm_tags for t in ["api", "reference", "endpoint", "swagger", "openapi"]):
        return "api_ref"
    if any(t in fm_tags for t in ["tutorial", "how-to", "walkthrough"]):
        return "tutorial"
    if any(t in fm_tags for t in ["concept", "overview", "introduction", "architecture"]):
        return "concept"
    if any(t in fm_tags for t in ["business", "process", "workflow", "domain"]):
        return "business"
    if any(t in fm_tags for t in ["guide", "getting-started"]):
        return "guide"

    # Detect from path
    for keyword, dtype in [
        ("api", "api_ref"),
        ("tutorial", "tutorial"),
        ("guide", "guide"),
        ("concept", "concept"),
        ("architecture", "concept"),
        ("business", "business"),
        ("process", "business"),
        ("getting-started", "guide"),
        ("reference", "api_ref"),
    ]:
        if keyword in p:
            return dtype

    return "doc"


# ── Frontmatter parsing ───────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from a markdown/MDX file.
    Returns (frontmatter_dict, body_without_frontmatter).
    Tolerates files with no frontmatter.
    """
    try:
        post = fm.loads(content)
        return dict(post.metadata), post.content
    except Exception as exc:
        log.debug("Frontmatter parse error (ignored): %s", exc)
        return {}, content


def extract_title(frontmatter: dict, body: str, path: str) -> str:
    """
    Resolve document title with priority:
    frontmatter.title > frontmatter.sidebar_label > first H1 in body > filename stem
    """
    if frontmatter.get("title"):
        return str(frontmatter["title"]).strip()
    if frontmatter.get("sidebar_label"):
        return str(frontmatter["sidebar_label"]).strip()

    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()

    return Path(path).stem.replace("-", " ").replace("_", " ").title()


def extract_description(frontmatter: dict, body: str) -> str:
    """
    Resolve description with priority:
    frontmatter.description > first non-heading paragraph
    """
    if frontmatter.get("description"):
        return str(frontmatter["description"]).strip()

    # First paragraph (skip headings, frontmatter fences, empty lines)
    lines = body.splitlines()
    paragraph: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("---"):
            if paragraph:
                break
            continue
        paragraph.append(stripped)

    return " ".join(paragraph)[:300]


def extract_tags(frontmatter: dict) -> list[str]:
    """Normalize frontmatter tags to a flat lowercase list."""
    raw = frontmatter.get("tags") or []
    if isinstance(raw, str):
        raw = [t.strip() for t in raw.split(",")]
    return [str(t).strip().lower() for t in raw if t]


# ── Image reference extraction ────────────────────────────────────────────────

_IMG_MD_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)',
    re.MULTILINE,
)
_IMG_JSX_RE = re.compile(
    r'<img\s[^>]*src=["\'](?P<src>[^"\']+)["\'][^>]*(?:alt=["\'](?P<alt>[^"\']*)["\'])?',
    re.IGNORECASE,
)


def extract_image_refs(content: str, doc_url: str, repo: str, branch: str) -> list[ImageRef]:
    """
    Find all image references in a markdown/MDX file.
    Collects alt text and the surrounding sentence/paragraph as context.
    """
    refs: list[ImageRef] = []
    lines = content.splitlines()

    for match in list(_IMG_MD_RE.finditer(content)) + list(_IMG_JSX_RE.finditer(content)):
        src = match.group("src").strip()
        alt = (match.group("alt") or "").strip()

        # Skip external images (http/https)
        if src.startswith("http"):
            continue

        # Find line number for surrounding context
        pos = match.start()
        line_no = content[:pos].count("\n")
        ctx_lines = lines[max(0, line_no - 2): line_no + 3]
        surrounding = " ".join(l.strip() for l in ctx_lines if l.strip())
        # Remove the image link itself from surrounding text
        surrounding = _IMG_MD_RE.sub("", surrounding).strip()

        # Attempt to build a github URL for the image
        img_path = src.lstrip("./").lstrip("/")
        img_url = f"https://github.com/{repo}/blob/{branch}/{img_path}"

        refs.append(ImageRef(
            path=img_path,
            url=img_url,
            alt_text=alt,
            surrounding_text=surrounding[:400],
        ))

    return refs


# ── Sidebar / category resolution ────────────────────────────────────────────

def resolve_category(path: str, sidebar_data: dict | None) -> str:
    """
    Given a file path and optional parsed sidebar data,
    return the Docusaurus category label (e.g. "Guides", "API Reference").
    Falls back to the immediate parent folder name.
    """
    if sidebar_data:
        # Try to find the doc's path in the sidebar tree
        label = _search_sidebar(path, sidebar_data)
        if label:
            return label

    # Fallback: use parent directory
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        folder = parts[-2]
        # Strip numeric prefixes like "01-getting-started"
        folder = re.sub(r"^\d+[-_]", "", folder)
        return folder.replace("-", " ").replace("_", " ").title()

    return ""


def _search_sidebar(doc_path: str, sidebar: dict, _label: str = "") -> str:
    """
    Recursively search sidebar JSON/JS for the category containing doc_path.
    sidebar JSON expected format: { "category": { "label": "...", "items": [...] } }
    """
    for _key, value in (sidebar.items() if isinstance(sidebar, dict) else []):
        if isinstance(value, dict):
            label = str(value.get("label", ""))
            items = value.get("items", [])
            for item in items:
                if isinstance(item, str) and doc_path.replace(".md", "").replace(".mdx", "") in item:
                    return label
                if isinstance(item, dict):
                    found = _search_sidebar(doc_path, item, label)
                    if found:
                        return found
    return ""


def parse_sidebar_file(content: str) -> dict:
    """
    Attempt to parse sidebars.json content.
    Returns empty dict on failure (sidebar is optional).
    """
    try:
        return json.loads(content)
    except Exception:
        # Try stripping JS module.exports
        try:
            stripped = re.sub(r"^module\.exports\s*=\s*", "", content.strip()).rstrip(";")
            return json.loads(stripped)
        except Exception:
            return {}
