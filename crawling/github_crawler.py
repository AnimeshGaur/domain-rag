"""
crawling/github_crawler.py
───────────────────────────
Docusaurus-aware GitHub crawler.

Crawls one or more GitHub repositories and returns RawDoc objects,
with full Docusaurus metadata (frontmatter, doc_type, category, images).

Key improvements over the original:
  • Handles both text files (md, mdx, code) and image files
  • Passes content through docusaurus_parser for rich metadata
  • Loads sidebars.json for category resolution
  • Respects rate limits with tenacity-based retry
  • Produces synthetic image text-docs via image_processor
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Generator

from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import cfg
from processing.doc_schema import RawDoc, ImageRef
from crawling.docusaurus_parser import (
    detect_section,
    detect_doc_type,
    parse_frontmatter,
    extract_title,
    extract_tags,
    extract_image_refs,
    resolve_category,
    parse_sidebar_file,
)
from crawling.image_processor import is_image, image_to_text_doc

log = logging.getLogger(__name__)


# ── Rate-limit-aware blob fetcher ─────────────────────────────────────────────

def _fetch_blob_with_retry(repo: Repository, sha: str, gh: Github) -> bytes | None:
    """Fetch a raw blob, sleeping through rate-limit resets."""
    for _ in range(5):
        try:
            blob = repo.get_git_blob(sha)
            if blob.encoding != "base64":
                return None
            return base64.b64decode(blob.content)
        except RateLimitExceededException:
            reset = gh.get_rate_limit().core.reset
            sleep_s = max(reset - time.time(), 0) + 5
            log.warning("GitHub rate limited — sleeping %.0fs", sleep_s)
            time.sleep(sleep_s)
        except GithubException as exc:
            log.warning("Blob fetch error (sha=%s): %s", sha[:8], exc)
            return None
    return None


# ── Per-repo crawl ────────────────────────────────────────────────────────────

def _crawl_repo(gh: Github, repo_slug: str) -> list[RawDoc]:
    """
    Walk the entire git tree of a repo and return RawDoc objects.
    Images are returned as synthetic text docs if they have surrounding context.
    """
    log.info("Crawling repo: %s", repo_slug)
    repo: Repository = gh.get_repo(repo_slug)
    branch = repo.default_branch

    # Collect all tree items in one API call
    tree = repo.get_git_tree(branch, recursive=True)

    # Load sidebar data if available (for category resolution)
    sidebar_data: dict = {}
    sidebar_paths = ["sidebars.js", "sidebars.json", "website/sidebars.js"]
    for sp in sidebar_paths:
        try:
            f = repo.get_contents(sp, ref=branch)
            raw = base64.b64decode(f.content).decode("utf-8")
            sidebar_data = parse_sidebar_file(raw)
            if sidebar_data:
                log.info("Loaded sidebar from %s", sp)
                break
        except Exception:
            pass

    text_docs: list[RawDoc] = []
    image_paths: list[tuple[str, str]] = []  # (path, sha)

    for item in tree.tree:
        if item.type != "blob":
            continue

        path = item.path
        ext = Path(path).suffix.lower()

        # ── Image files ──────────────────────────────────────────────
        if cfg.GITHUB_INCLUDE_IMAGES and is_image(path):
            img_url = f"https://github.com/{repo_slug}/blob/{branch}/{path}"
            image_paths.append((path, img_url))
            continue

        # ── Text files ───────────────────────────────────────────────
        if ext not in cfg.GITHUB_TEXT_EXTS:
            continue

        raw_bytes = _fetch_blob_with_retry(repo, item.sha, gh)
        if raw_bytes is None:
            continue
        if len(raw_bytes) > cfg.GITHUB_MAX_FILE_SIZE_KB * 1024:
            log.debug("Skipping oversized file: %s", path)
            continue
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            log.debug("Skipping binary/non-UTF8 file: %s", path)
            continue

        # Parse Docusaurus metadata
        frontmatter, body = parse_frontmatter(content) if ext in (".md", ".mdx") else ({}, content)
        section = detect_section(path, cfg.DOCUSAURUS_DOCS_PATH)
        doc_type = detect_doc_type(path, frontmatter)

        # Skip blog posts if not configured to include them
        if doc_type == "blog" and not cfg.DOCUSAURUS_INCLUDE_BLOG:
            continue

        category = resolve_category(path, sidebar_data)
        tags = extract_tags(frontmatter)
        image_refs = extract_image_refs(body, f"https://github.com/{repo_slug}/blob/{branch}/{path}", repo_slug, branch)

        text_docs.append(RawDoc(
            repo=repo_slug,
            path=path,
            content=content,
            sha=item.sha,
            url=f"https://github.com/{repo_slug}/blob/{branch}/{path}",
            branch=branch,
            doc_type=doc_type,
            section=section,
            category=category,
            frontmatter=frontmatter,
            images=image_refs,
        ))

    log.info(
        "Repo %s: %d text files, %d images discovered",
        repo_slug, len(text_docs), len(image_paths),
    )

    # ── Synthetic image docs ──────────────────────────────────────────
    if cfg.GITHUB_INCLUDE_IMAGES and image_paths:
        for img_path, img_url in image_paths:
            img_doc = image_to_text_doc(
                image_path=img_path,
                image_url=img_url,
                repo=repo_slug,
                branch=branch,
                referring_docs=text_docs,
            )
            if img_doc:
                text_docs.append(img_doc)

    return text_docs


def _sanitize_slug(slug: str) -> str:
    """Extract 'owner/repo' from URLs or messy strings."""
    slug = slug.strip()
    # Remove protocol
    if "://" in slug:
        slug = slug.split("://")[-1]
    # Remove domain (github.com/)
    if slug.lower().startswith("github.com/"):
        slug = slug[11:]
    # Remove .git suffix
    if slug.lower().endswith(".git"):
        slug = slug[:-4]
    # If it's a deep URL like .../blob/main/..., take first two parts
    parts = [p for p in slug.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return slug


def crawl_repos(repo_slugs: list[str] | None = None) -> list[RawDoc]:
    """
    Crawl one or more GitHub repositories.

    Args:
        repo_slugs: List of "owner/repo" strings. Defaults to cfg.GITHUB_REPOS.

    Returns:
        Flat list of RawDoc objects ready for processing.
    """
    gh = Github()
    slugs = repo_slugs or cfg.GITHUB_REPOS
    all_docs: list[RawDoc] = []

    for raw_slug in slugs:
        slug = _sanitize_slug(raw_slug)
        if not slug:
            continue
        try:
            docs = _crawl_repo(gh, slug)
            all_docs.extend(docs)
        except Exception as exc:
            log.error("Failed to crawl %s: %s", slug, exc)

    log.info(
        "Total crawled: %d documents across %d repo(s)",
        len(all_docs), len(slugs),
    )
    return all_docs
