"""
llm/artifact_generator.py
──────────────────────────
Given a set of ProcessedDocs (from one logical module / section),
calls an LLM to generate enriched artifact documents.

Docusaurus artifact types (Phase 2 only):
  • component_doc    — component responsibilities, interface, dependencies
  • api_contract     — HTTP endpoints, request/response schemas
  • sequence_flow    — runtime interaction sequences
  • data_flow        — data movement and transformations
  • arch_summary     — high-level architecture overview
  • business_process — business workflow, actors, decision points
  • domain_model     — domain entities, glossary, relationships
  • user_guide       — synthesized onboarding guide for end users

These artifacts are passed to the chunker and indexed identically to
raw docs. The LLM layer is transparent to the search layer.
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Literal

from processing.doc_schema import ProcessedDoc
from llm.base import LLMProvider

log = logging.getLogger(__name__)

ArtifactType = Literal[
    "component_doc",
    "api_contract",
    "sequence_flow",
    "data_flow",
    "arch_summary",
    "business_process",
    "domain_model",
    "user_guide",
]


@dataclass
class LLMArtifact:
    """
    An LLM-generated artifact document.
    Structurally equivalent to ProcessedDoc for downstream chunking.
    """
    artifact_type: ArtifactType
    title: str
    content: str                  # generated Markdown
    repo: str
    source_paths: list[str]
    source_urls: list[str]
    category: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = textwrap.dedent("""
    You are an expert technical writer and software architect.
    You receive Docusaurus documentation pages from a GitHub repository
    and produce precise, structured Markdown artifacts.

    Rules:
    - Be concise but complete. Use headings, tables, bullet lists, and code blocks.
    - Never invent information not present in the source documents.
    - Preserve technical accuracy. Use exact names, types, and values from the source.
    - If information is missing, say "Not documented" rather than guessing.
    - Format output as clean Markdown with a single H1 title at the top.
""").strip()

# ── Per-type prompts ──────────────────────────────────────────────────────────

_PROMPTS: dict[str, str] = {
    "component_doc": textwrap.dedent("""
        Generate a **Component Documentation** artifact.
        Structure:
        # [Component Name] — Component Documentation
        ## Overview
        ## Responsibilities
        ## Public Interface (classes, functions, hooks, exports)
        ## Inputs & Outputs / Props / Parameters
        ## Internal Dependencies
        ## External Dependencies
        ## Configuration Options
        ## Known Limitations / TODOs
    """).strip(),

    "api_contract": textwrap.dedent("""
        Generate an **API Contract** artifact covering all HTTP endpoints,
        RPC methods, or message schemas found in the documents.
        Structure:
        # API Contract
        ## Base URL
        ## Authentication
        ## Endpoints
        For each: Method, Path, Description, Request Schema, Response Schema, Error Codes
        ## Shared Types / DTOs
        ## Breaking Change Notes
    """).strip(),

    "sequence_flow": textwrap.dedent("""
        Generate a **Sequence Flow** artifact describing runtime interactions.
        Structure:
        # [Flow Name] — Sequence Flow
        ## Actors / Components Involved
        ## Preconditions
        ## Step-by-Step Sequence (numbered)
        ## Error / Exception Paths
        ## Async / Timing Notes
        ## Diagram (Mermaid sequence if data supports it)
    """).strip(),

    "data_flow": textwrap.dedent("""
        Generate a **Data Flow** artifact.
        Structure:
        # Data Flow
        ## Data Sources
        ## Transformations & Processing Steps
        ## Storage (schemas, indexes, tables)
        ## Data Lineage (input → process → output)
        ## Retention & Privacy Notes
    """).strip(),

    "arch_summary": textwrap.dedent("""
        Generate an **Architecture Summary** artifact.
        Structure:
        # Architecture Summary
        ## Purpose & Scope
        ## Architecture Pattern (layered / event-driven / microservices / …)
        ## Key Components & their Roles
        ## Key Design Decisions & Rationale
        ## Scalability & Performance Considerations
        ## Security Boundaries
        ## Deployment Notes
    """).strip(),

    "business_process": textwrap.dedent("""
        Generate a **Business Process** artifact.
        Structure:
        # [Process Name] — Business Process
        ## Purpose & Business Goal
        ## Stakeholders & Actors
        ## Triggers / Entry Points
        ## Process Steps (numbered, with decision points)
        ## Expected Outcomes
        ## Exceptions & Edge Cases
        ## SLA / Timing Considerations
    """).strip(),

    "domain_model": textwrap.dedent("""
        Generate a **Domain Model** artifact.
        Structure:
        # Domain Model
        ## Core Entities (table: Name | Description | Key Attributes)
        ## Relationships Between Entities
        ## Glossary (term definitions)
        ## Business Rules & Invariants
        ## Bounded Contexts (if applicable)
    """).strip(),

    "user_guide": textwrap.dedent("""
        Generate a **User Guide** artifact for end users or developers
        who are new to this system.
        Structure:
        # Getting Started Guide
        ## What Is This?
        ## Prerequisites
        ## Quick Start (step-by-step)
        ## Key Concepts to Understand First
        ## Common Operations (with examples)
        ## Troubleshooting
        ## Where to Get Help
    """).strip(),
}


# ── Auto-detection ────────────────────────────────────────────────────────────

def _detect_artifact_types(docs: list[ProcessedDoc]) -> list[ArtifactType]:
    """
    Heuristically determine which artifact types to generate for a set of docs.
    Always generates component_doc + arch_summary.
    Additional types triggered by content signals.
    """
    types: list[ArtifactType] = ["component_doc", "arch_summary"]

    combined = " ".join(
        " ".join([d.clean_text[:800], d.title, " ".join(d.tags)])
        for d in docs[:5]
    ).lower()

    signals: dict[str, list[str]] = {
        "api_contract": ["endpoint", "route", "rest", "graphql", "api", "http", "post", "get"],
        "sequence_flow": ["flow", "sequence", "async", "event", "queue", "step", "then"],
        "data_flow": ["data", "schema", "model", "database", "migration", "transform"],
        "business_process": ["workflow", "process", "approval", "stakeholder", "business"],
        "domain_model": ["entity", "domain", "glossary", "concept", "definition"],
        "user_guide": ["getting started", "quickstart", "install", "setup", "tutorial"],
    }

    for atype, keywords in signals.items():
        if any(kw in combined for kw in keywords):
            if atype not in types:
                types.append(atype)  # type: ignore[arg-type]

    return types  # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_artifacts(
    docs: list[ProcessedDoc],
    llm: LLMProvider,
    artifact_types: list[str] | None = None,
) -> list[LLMArtifact]:
    """
    Generate structured artifacts from a group of related ProcessedDocs using an LLM.

    Args:
        docs:           Related documents (e.g. all docs in one sidebar category).
        llm:            LLM provider to use for generation.
        artifact_types: Override auto-detected types.

    Returns:
        List of LLMArtifact ready for chunking.
    """
    if not docs:
        return []

    types_to_gen = artifact_types or _detect_artifact_types(docs)
    artifacts: list[LLMArtifact] = []

    # Build source content block (trimmed for context window)
    source_block = "\n\n---\n\n".join(
        f"### {d.title} ({d.path})\n\n{d.clean_text[:3000]}"
        for d in docs[:15]  # cap to 15 docs per LLM call
    )

    for art_type in types_to_gen:
        if art_type not in _PROMPTS:
            log.warning("Unknown artifact type %r — skipping", art_type)
            continue

        user_msg = f"{_PROMPTS[art_type]}\n\n---\n\n# Source Documents\n\n{source_block}"
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

        log.info(
            "Generating %s for %d docs from %s",
            art_type, len(docs), docs[0].repo,
        )

        try:
            content = llm.complete(messages, temperature=0.2, max_tokens=2048)
        except Exception as exc:
            log.error("LLM generation failed for %s: %s", art_type, exc)
            continue

        # Extract title from first H1 or fall back to type name
        title = art_type.replace("_", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        artifacts.append(LLMArtifact(
            artifact_type=art_type,  # type: ignore[arg-type]
            title=title,
            content=content,
            repo=docs[0].repo,
            source_paths=[d.path for d in docs],
            source_urls=[d.url for d in docs],
            category=docs[0].category,
            tags=list({t for d in docs for t in d.tags}),
            metadata={
                "source_doc_count": len(docs),
                "branch": docs[0].branch,
            },
        ))

    return artifacts


def group_docs_by_category(docs: list[ProcessedDoc]) -> dict[str, list[ProcessedDoc]]:
    """Group docs by Docusaurus category for batched artifact generation."""
    groups: dict[str, list[ProcessedDoc]] = {}
    for doc in docs:
        key = doc.category or doc.section or "__root__"
        groups.setdefault(key, []).append(doc)
    return groups
