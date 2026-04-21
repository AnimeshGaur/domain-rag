"""
ingestion/pipeline.py
──────────────────────
Orchestrates the full ingestion pipeline in two modes:

  Mode A — No-LLM (Phase 1, default):
    GitHub crawl → Docusaurus parse → Markdown process → Chunk → Embed → Index
    Requires: GITHUB_TOKEN only

  Mode B — LLM (Phase 2):
    GitHub crawl → Docusaurus parse → LLM artifact generation → Chunk → Embed → Index
    Requires: GITHUB_TOKEN + OPENAI_API_KEY (or whichever LLM provider)

Both modes produce the same Elasticsearch index schema.
You can switch from Mode A to Mode B without re-tooling the search layer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import cfg
from crawling.github_crawler import crawl_repos
from processing.markdown_processor import process_docs
from processing.chunker import chunk_docs, chunk_doc
from processing.doc_schema import Chunk, ProcessedDoc
from embedding.base import get_embed_provider
from search import elastic_store

log = logging.getLogger(__name__)


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    repos: list[str]
    mode: str = "no-llm"            # "no-llm" | "llm"
    docs_crawled: int = 0
    docs_processed: int = 0
    artifacts_generated: int = 0    # only in LLM mode
    chunks_produced: int = 0
    chunks_indexed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── Ingestion state (for /status polling) ─────────────────────────────────────

class _IngestionState:
    def __init__(self) -> None:
        self.running: bool = False
        self.last_result: PipelineResult | None = None

    def start(self) -> None:
        self.running = True

    def finish(self, result: PipelineResult) -> None:
        self.running = False
        self.last_result = result

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "last_result": _result_to_dict(self.last_result) if self.last_result else None,
        }


_state = _IngestionState()


def get_ingestion_state() -> _IngestionState:
    return _state


def _result_to_dict(r: PipelineResult) -> dict:
    return {
        "repos": r.repos,
        "mode": r.mode,
        "docs_crawled": r.docs_crawled,
        "docs_processed": r.docs_processed,
        "artifacts_generated": r.artifacts_generated,
        "chunks_produced": r.chunks_produced,
        "chunks_indexed": r.chunks_indexed,
        "duration_seconds": r.duration_seconds,
        "errors": r.errors,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_ingestion(
    repos: list[str] | None = None,
    full_reset: bool = False,
    use_llm: bool | None = None,
    artifact_types: list[str] | None = None,
) -> PipelineResult:
    """
    Run the ingestion pipeline.

    Args:
        repos:          "owner/repo" list. Defaults to cfg.GITHUB_REPOS.
        full_reset:     Drop + recreate the index before ingesting.
        use_llm:        True = LLM mode, False = no-LLM mode.
                        Defaults to cfg.INGEST_USE_LLM.
        artifact_types: Override auto-detected LLM artifact types.

    Returns:
        PipelineResult with stats.
    """
    t0 = time.time()
    mode_use_llm = use_llm if use_llm is not None else cfg.INGEST_USE_LLM
    mode = "llm" if mode_use_llm else "no-llm"
    result = PipelineResult(repos=repos or cfg.GITHUB_REPOS, mode=mode)

    _state.start()

    try:
        embed_provider = get_embed_provider()
        log.info(
            "Pipeline start | mode=%s | embed=%s(%s) | repos=%s",
            mode, cfg.EMBED_PROVIDER, embed_provider.model_name, repos or cfg.GITHUB_REPOS,
        )

        # ── 0. Prepare index ──────────────────────────────────────────────────
        if full_reset:
            log.info("Full reset: deleting index …")
            elastic_store.recreate_index(embed_dims=embed_provider.dims)
        else:
            elastic_store.ensure_index(embed_dims=embed_provider.dims)

        # ── 1. Crawl ──────────────────────────────────────────────────────────
        log.info("Step 1: Crawling GitHub repos …")
        raw_docs = crawl_repos(repos)
        result.docs_crawled = len(raw_docs)

        if not raw_docs:
            result.errors.append("No documents crawled — check GITHUB_REPOS and GITHUB_TOKEN.")
            return result

        # ── 2. Process (parse markdown, strip MDX, extract metadata) ──────────
        log.info("Step 2: Processing %d docs …", len(raw_docs))
        processed_docs = process_docs(raw_docs)
        result.docs_processed = len(processed_docs)

        # ── 3a. Mode A: chunk directly from processed docs (no LLM) ───────────
        if not mode_use_llm:
            log.info("Step 3 [no-llm]: Chunking %d docs …", len(processed_docs))
            chunks = chunk_docs(processed_docs)
            result.chunks_produced = len(chunks)

        # ── 3b. Mode B: generate LLM artifacts, then chunk ────────────────────
        else:
            if not cfg.llm_enabled:
                msg = (
                    "use_llm=True but LLM is not configured "
                    "(set LLM_PROVIDER=openai and OPENAI_API_KEY). "
                    "Falling back to no-llm mode."
                )
                log.warning(msg)
                result.errors.append(msg)
                chunks = chunk_docs(processed_docs)
                result.chunks_produced = len(chunks)
            else:
                log.info("Step 3 [llm]: Generating artifacts …")
                chunks = _run_llm_ingestion(
                    processed_docs, artifact_types, result
                )

        # ── 4. Embed ──────────────────────────────────────────────────────────
        # Parent chunks provide LLM context — stored in ES but NOT embedded for kNN
        child_chunks = [c for c in chunks if not c.is_parent]
        parent_chunks = [c for c in chunks if c.is_parent]
        log.info(
            "Step 4: Embedding %d child chunks (skipping %d parent) with %s …",
            len(child_chunks), len(parent_chunks), embed_provider.model_name,
        )
        texts = [c.text for c in child_chunks]
        vectors = embed_provider.embed_texts(texts)
        child_vec_pairs = list(zip(child_chunks, vectors))

        # Parent chunks use a zero vector — ES stores them, kNN filter (is_parent=false) excludes them
        zero_vec = [0.0] * embed_provider.dims
        parent_vec_pairs = [(c, zero_vec) for c in parent_chunks]
        chunk_vec_pairs = child_vec_pairs + parent_vec_pairs

        # ── 5. Index ──────────────────────────────────────────────────────────
        log.info("Step 5: Indexing %d total chunks (child + parent) …", len(chunk_vec_pairs))
        result.chunks_indexed = elastic_store.upsert_chunks(chunk_vec_pairs)

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        result.errors.append(str(exc))
    finally:
        result.duration_seconds = round(time.time() - t0, 2)
        _state.finish(result)

    log.info(
        "Pipeline complete | mode=%s | docs=%d | chunks=%d | indexed=%d | %.1fs",
        mode, result.docs_crawled, result.chunks_produced,
        result.chunks_indexed, result.duration_seconds,
    )
    return result


def _run_llm_ingestion(
    processed_docs: list[ProcessedDoc],
    artifact_types: list[str] | None,
    result: PipelineResult,
) -> list[Chunk]:
    """Run LLM artifact generation and chunk the resulting artifacts."""
    from llm.base import get_llm_provider
    from llm.artifact_generator import generate_artifacts, group_docs_by_category
    from processing.doc_schema import ProcessedDoc as PD

    llm_provider = get_llm_provider()
    groups = group_docs_by_category(processed_docs)

    all_chunks: list[Chunk] = []
    all_artifact_count = 0

    for group_key, group_docs in groups.items():
        log.info("Generating artifacts for category '%s' (%d docs) …", group_key, len(group_docs))
        try:
            artifacts = generate_artifacts(group_docs, llm_provider, artifact_types)
            all_artifact_count += len(artifacts)

            for artifact in artifacts:
                # Convert LLMArtifact → ProcessedDoc-compatible for chunking
                pseudo_doc = PD(
                    repo=artifact.repo,
                    path=f"__artifact__/{artifact.artifact_type}/{group_key}",
                    url=artifact.source_urls[0] if artifact.source_urls else "",
                    sha="",
                    branch="",
                    doc_type=artifact.artifact_type,
                    section="docs",
                    category=artifact.category,
                    frontmatter={},
                    images=[],
                    title=artifact.title,
                    description="",
                    clean_text=artifact.content,
                    heading_tree=[],
                    tags=artifact.tags,
                    links=[],
                )
                chunks = chunk_doc(pseudo_doc, artifact_type=artifact.artifact_type)
                all_chunks.extend(chunks)

        except Exception as exc:
            msg = f"Artifact gen failed for '{group_key}': {exc}"
            log.error(msg)
            result.errors.append(msg)

    result.artifacts_generated = all_artifact_count
    result.chunks_produced = len(all_chunks)
    return all_chunks
