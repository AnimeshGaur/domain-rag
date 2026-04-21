"""
cli/ingest.py
──────────────
CLI entry point for the ingestion pipeline.

Usage:
  # Phase 1 — No LLM (default):
  python -m cli.ingest --repos your-org/your-repo

  # Phase 2 — With LLM artifact generation:
  python -m cli.ingest --repos your-org/your-repo --llm

  # Full reset + specific artifact types:
  python -m cli.ingest --full-reset --llm --artifact-types component_doc api_contract
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGbase ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repos", nargs="+", metavar="OWNER/REPO",
        help="GitHub repos to ingest (overrides GITHUB_REPOS env var)",
    )
    parser.add_argument(
        "--full-reset", action="store_true",
        help="Drop and recreate the Elasticsearch index before ingesting",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-llm", action="store_true", default=False,
        help="Phase 1: index raw docs directly (no LLM calls, default)",
    )
    mode_group.add_argument(
        "--llm", action="store_true", default=False,
        help="Phase 2: use LLM to generate enriched artifacts before indexing",
    )

    parser.add_argument(
        "--artifact-types", nargs="+",
        metavar="TYPE",
        help="Override artifact types for LLM mode "
             "(e.g. component_doc api_contract business_process)",
    )

    args = parser.parse_args()

    # Determine mode
    use_llm: bool | None = None
    if args.llm:
        use_llm = True
    elif args.no_llm:
        use_llm = False
    # else: None → use cfg.INGEST_USE_LLM

    from ingestion.pipeline import run_ingestion

    log.info("Starting ingestion | mode=%s | repos=%s", "llm" if use_llm else "no-llm", args.repos)
    result = run_ingestion(
        repos=args.repos,
        full_reset=args.full_reset,
        use_llm=use_llm,
        artifact_types=args.artifact_types,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Ingestion Complete — {result.mode.upper()} mode")
    print("=" * 60)
    print(f"  Docs crawled:       {result.docs_crawled}")
    print(f"  Docs processed:     {result.docs_processed}")
    if result.mode == "llm":
        print(f"  Artifacts generated:{result.artifacts_generated}")
    print(f"  Chunks produced:    {result.chunks_produced}")
    print(f"  Chunks indexed:     {result.chunks_indexed}")
    print(f"  Duration:           {result.duration_seconds}s")
    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"    - {e}")
    print("=" * 60)

    sys.exit(0 if not result.errors else 1)


if __name__ == "__main__":
    main()
