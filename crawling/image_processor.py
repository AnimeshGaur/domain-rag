"""
crawling/image_processor.py
────────────────────────────
Processes image files crawled from GitHub:
  • Extracts alt-text and captions from surrounding markdown
  • Converts image metadata into searchable text documents (RawDoc)
  • Allows images to be discovered via semantic search

Images themselves cannot be embedded as dense vectors without a
vision model. Instead we index their textual metadata so that
queries like "show me the auth flow diagram" can surface them.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from processing.doc_schema import RawDoc, ImageRef

log = logging.getLogger(__name__)

# Image extensions we handle
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".bmp", ".ico"}


def is_image(path: str) -> bool:
    """Return True if the file path has an image extension."""
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return suffix in IMAGE_EXTS


def image_to_text_doc(
    image_path: str,
    image_url: str,
    repo: str,
    branch: str,
    referring_docs: list[RawDoc],
) -> Optional[RawDoc]:
    """
    Build a synthetic text RawDoc for an image file by aggregating
    alt-text and captions collected from all documents that reference it.

    Returns None if no textual metadata is found (image is unreferenced).
    """
    alt_texts: list[str] = []
    captions: list[str] = []
    surrounding_texts: list[str] = []
    categories: set[str] = set()

    for doc in referring_docs:
        for img_ref in doc.images:
            # Match by filename suffix to handle relative paths
            if _paths_match(img_ref.path, image_path):
                if img_ref.alt_text:
                    alt_texts.append(img_ref.alt_text)
                if img_ref.caption:
                    captions.append(img_ref.caption)
                if img_ref.surrounding_text:
                    surrounding_texts.append(img_ref.surrounding_text)
                if doc.category:
                    categories.add(doc.category)

    if not any([alt_texts, captions, surrounding_texts]):
        log.debug("No textual metadata for image %s — skipping indexing", image_path)
        return None

    # Build synthetic markdown content for the image
    img_name = image_path.rsplit("/", 1)[-1]
    parts = [f"# Image: {img_name}\n"]
    if alt_texts:
        parts.append("## Alt Text\n" + "\n".join(f"- {t}" for t in alt_texts))
    if captions:
        parts.append("## Captions\n" + "\n".join(f"- {c}" for c in captions))
    if surrounding_texts:
        parts.append("## Context\n" + "\n\n".join(surrounding_texts))

    content = "\n\n".join(parts)

    return RawDoc(
        repo=repo,
        path=image_path,
        content=content,
        sha="",
        url=image_url,
        branch=branch,
        doc_type="image",
        section="static",
        category=", ".join(sorted(categories)),
        frontmatter={
            "title": f"Image: {img_name}",
            "tags": ["image", "diagram", "figure"],
        },
        images=[ImageRef(
            path=image_path,
            url=image_url,
            alt_text="; ".join(alt_texts),
        )]
    )


def _paths_match(ref_path: str, image_path: str) -> bool:
    """Check if two image paths refer to the same file (handles relative vs full paths)."""
    def _normalize(p: str) -> str:
        return p.replace("\\", "/").lstrip("./").lower()

    rn, ip = _normalize(ref_path), _normalize(image_path)
    # Match if either is a suffix of the other
    return rn == ip or rn.endswith("/" + ip) or ip.endswith("/" + rn)
