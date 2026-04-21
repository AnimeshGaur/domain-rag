"""
cli/query.py
─────────────
CLI query tool — works with and without LLM.

Usage:
  # Search mode (Phase 1 — no LLM):
  python -m cli.query --search "What is the authentication flow?"

  # Answer mode (Phase 2 — LLM required):
  python -m cli.query "What is the authentication flow?"

  # Interactive mode:
  python -m cli.query --interactive

  # With filters:
  python -m cli.query --filter doc_type=api_ref "List all endpoints"
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.WARNING,     # suppress library noise in CLI
    format="%(levelname)s | %(message)s",
)


def _parse_filter(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Filter must be key=value, got: {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def _print_search_results(resp) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Search results for: {resp.question!r}")
    print(f"  {resp.retrieval_hits} candidates → {resp.reranked_to} after rerank")
    print(f"{'─' * 60}")
    for i, r in enumerate(resp.results, 1):
        badge = f"[{r.doc_type}:{r.category or '—'}]"
        print(f"\n  {i}. {r.title} {badge}")
        print(f"     Score: {r.score:.4f} | Rerank: {r.rerank_score or '—'}")
        print(f"     {r.heading_path}")
        print(f"     {r.url}")
        snippet = r.text[:200].replace("\n", " ")
        print(f"     …{snippet}…")
    print("")


def _print_answer(result) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Answer")
    print(f"{'═' * 60}")
    print(result.answer)
    if result.sources:
        print(f"\n{'─' * 60}")
        print(f"  Sources ({len(result.sources)})")
        print(f"{'─' * 60}")
        for i, s in enumerate(result.sources, 1):
            print(f"  [{i}] {s.title} [{s.artifact_type}]")
            print(f"       {s.url}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGbase query CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="?", help="Question or search query")
    parser.add_argument(
        "--search", "-s", action="store_true",
        help="Search mode: return ranked chunks without LLM answer",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode: continuous question loop",
    )
    parser.add_argument(
        "--filter", "-f", dest="filters", type=_parse_filter, action="append",
        metavar="KEY=VALUE",
        help="Metadata filter (can be used multiple times)",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)

    args = parser.parse_args()
    filters = dict(args.filters) if args.filters else None

    from retrieval.query_engine import search, answer
    from llm.base import LLMNotEnabledError

    def run_query(question: str) -> None:
        if args.search:
            resp = search(question, filters=filters, top_k=args.top_k, rerank_top_n=args.top_n)
            _print_search_results(resp)
        else:
            try:
                result = answer(question, filters=filters)
                _print_answer(result)
            except LLMNotEnabledError:
                print(
                    "\n⚠ LLM not configured. Falling back to search mode.\n"
                    "  Set LLM_PROVIDER=openai and OPENAI_API_KEY to enable answers.\n"
                )
                resp = search(question, filters=filters, top_k=args.top_k, rerank_top_n=args.top_n)
                _print_search_results(resp)

    if args.interactive:
        print("RAGbase Interactive Mode (Ctrl+C or 'exit' to quit)\n")
        while True:
            try:
                q = input("❯ ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                run_query(q)
    elif args.question:
        run_query(args.question)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
