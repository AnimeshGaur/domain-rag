"""
ingestion/github_crawler.py
────────────────────────────
Crawls one or more GitHub repositories and returns a flat list of
RawFile objects (path, content, metadata).

Uses PyGitHub (pip install PyGitHub).
Respects rate limits via built-in retry / sleep.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator

from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository

from config import cfg

log = logging.getLogger(__name__)


@dataclass
class RawFile:
    repo: str          # "owner/repo"
    path: str          # file path inside repo
    content: str       # decoded UTF-8 text
    sha: str           # blob SHA – used for dedup / change detection
    url: str           # HTML URL for citation links
    branch: str        # default branch name
    language: str = "" # inferred from extension


def _infer_language(path: str) -> str:
    ext_map = {
        ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
        ".js": "JavaScript", ".jsx": "JavaScript", ".go": "Go",
        ".java": "Java", ".yaml": "YAML", ".yml": "YAML",
        ".json": "JSON", ".md": "Markdown", ".toml": "TOML",
    }
    for ext, lang in ext_map.items():
        if path.endswith(ext):
            return lang
    return "Unknown"


def _should_include(path: str) -> bool:
    """Return True if the file extension is in the allow-list."""
    return any(path.endswith(ext) for ext in cfg.GITHUB_INCLUDE_EXTS)


def _crawl_repo(gh: Github, repo_slug: str) -> Generator[RawFile, None, None]:
    """Recursively walk a repo tree and yield qualifying files."""
    log.info("Crawling repo: %s", repo_slug)
    repo: Repository = gh.get_repo(repo_slug)
    branch = repo.default_branch

    # Walk the git tree in one API call – far more efficient than recursive get_contents
    tree = repo.get_git_tree(branch, recursive=True)
    for item in tree.tree:
        if item.type != "blob":
            continue
        if not _should_include(item.path):
            continue

        # Fetch blob content with retry on rate limit
        while True:
            try:
                blob = repo.get_git_blob(item.sha)
                break
            except RateLimitExceededException:
                reset = gh.get_rate_limit().core.reset
                sleep_s = max((reset - time.time()), 0) + 5
                log.warning("Rate limited – sleeping %.0fs", sleep_s)
                time.sleep(sleep_s)
            except GithubException as exc:
                log.warning("Skipping %s (%s)", item.path, exc)
                blob = None
                break

        if blob is None:
            continue
        if blob.encoding != "base64":
            log.debug("Skipping non-base64 blob: %s", item.path)
            continue

        import base64
        raw_bytes = base64.b64decode(blob.content)

        # Skip binary / huge files
        if len(raw_bytes) > cfg.GITHUB_MAX_FILE_SIZE_KB * 1024:
            log.debug("Skipping oversized file: %s", item.path)
            continue
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            log.debug("Skipping binary file: %s", item.path)
            continue

        yield RawFile(
            repo=repo_slug,
            path=item.path,
            content=content,
            sha=item.sha,
            url=f"https://github.com/{repo_slug}/blob/{branch}/{item.path}",
            branch=branch,
            language=_infer_language(item.path),
        )


def crawl_repos(repo_slugs: list[str] | None = None) -> list[RawFile]:
    """
    Main entry point.  Pass a list of "owner/repo" strings, or omit to
    use cfg.GITHUB_REPOS.
    """
    gh = Github(cfg.GITHUB_TOKEN)
    slugs = repo_slugs or cfg.GITHUB_REPOS
    files: list[RawFile] = []
    for slug in slugs:
        slug = slug.strip()
        if not slug:
            continue
        try:
            files.extend(_crawl_repo(gh, slug))
        except Exception as exc:
            log.error("Failed to crawl %s: %s", slug, exc)
    log.info("Crawled %d files across %d repos", len(files), len(slugs))
    return files
