"""
ingestion/artifact_generator.py
────────────────────────────────
For each batch of related raw files, calls OpenAI to produce one or
more structured "artifact" documents (component doc, API contract, etc.).

Artifacts are plain-text / Markdown blobs that will be chunked and
embedded downstream.
"""
from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI

from config import cfg
from ingestion.github_crawler import RawFile

log = logging.getLogger(__name__)
client = OpenAI(api_key=cfg.OPENAI_API_KEY)

ArtifactType = Literal[
    "component_doc",
    "api_contract",
    "sequence_flow",
    "data_flow",
    "arch_summary",
]


@dataclass
class Artifact:
    artifact_type: ArtifactType
    title: str
    content: str                    # Markdown text
    source_repo: str
    source_paths: list[str]         # which files contributed
    source_urls: list[str]          # GitHub HTML links
    language: str = ""
    metadata: dict = field(default_factory=dict)


# ── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM = textwrap.dedent("""
    You are an expert software architect and technical writer.
    You receive source code files from a GitHub repository and produce
    precise, structured documentation artifacts in Markdown.
    Be concise but complete.  Use headings, bullet lists, and code fences.
    Never invent information not present in the source.
""").strip()

_PROMPTS: dict[ArtifactType, str] = {
    "component_doc": textwrap.dedent("""
        Produce a **Component Documentation** artifact for the provided files.
        Structure:
        ## Overview
        ## Responsibilities
        ## Public Interface (classes / functions / exports)
        ## Inputs & Outputs
        ## Dependencies (internal + external)
        ## Configuration
        ## Known Limitations / TODOs
    """).strip(),

    "api_contract": textwrap.dedent("""
        Produce an **API Contract** artifact covering all HTTP endpoints,
        RPC methods, or message schemas found in the provided files.
        Structure:
        ## Endpoints / Methods
        For each: method, path/name, description, request schema, response schema,
        error codes, authentication requirements.
        ## Shared Types / DTOs
        ## Breaking-change notes
    """).strip(),

    "sequence_flow": textwrap.dedent("""
        Produce a **Sequence Flow** artifact describing the runtime interaction
        between components visible in the provided files.
        Structure:
        ## Flow Name
        ## Actors / Components
        ## Step-by-Step Sequence (numbered)
        ## Error Paths
        ## Timing / Async notes
    """).strip(),

    "data_flow": textwrap.dedent("""
        Produce a **Data Flow** artifact describing how data moves through
        the system based on the provided files.
        Structure:
        ## Data Sources
        ## Transformations
        ## Storage (schemas, indexes, collections)
        ## Data Lineage (input → process → output)
        ## Retention / Privacy notes
    """).strip(),

    "arch_summary": textwrap.dedent("""
        Produce an **Architecture Summary** artifact giving a high-level
        picture of what this module / service does and where it fits.
        Structure:
        ## Purpose
        ## Architecture Pattern (e.g. layered, event-driven, CQRS …)
        ## Key Design Decisions
        ## Scalability & Performance Considerations
        ## Security Boundaries
        ## Deployment Notes
    """).strip(),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_user_message(files: list[RawFile], artifact_type: ArtifactType) -> str:
    """Concatenate file contents into a single message, with fences."""
    parts = [_PROMPTS[artifact_type], "\n\n---\n# Source Files\n"]
    for f in files:
        fence = f"```{f.language.lower() or 'text'}"
        parts.append(f"### {f.path}  ({f.repo})\n{fence}\n{f.content}\n```\n")
    return "\n".join(parts)


def _detect_relevant_types(files: list[RawFile]) -> list[ArtifactType]:
    """Heuristically decide which artifact types make sense for this file set."""
    paths = " ".join(f.path for f in files).lower()
    content_sample = " ".join(f.content[:500] for f in files[:3]).lower()
    combined = paths + " " + content_sample

    types: list[ArtifactType] = ["component_doc", "arch_summary"]  # always

    api_signals = ["router", "endpoint", "route", "handler", "controller",
                   "@app.", "fastapi", "express", "flask", "grpc", "openapi"]
    if any(s in combined for s in api_signals):
        types.append("api_contract")

    flow_signals = ["async", "await", "queue", "event", "publish", "subscribe",
                    "kafka", "rabbitmq", "celery", "worker"]
    if any(s in combined for s in flow_signals):
        types.append("sequence_flow")

    data_signals = ["model", "schema", "migration", "database", "db.", "orm",
                    "sqlalchemy", "prisma", "mongoose", "entity"]
    if any(s in combined for s in data_signals):
        types.append("data_flow")

    return list(dict.fromkeys(types))   # preserve order, dedup


# ── Public API ────────────────────────────────────────────────────────────────

def generate_artifacts(
    files: list[RawFile],
    artifact_types: list[ArtifactType] | None = None,
) -> list[Artifact]:
    """
    Given a list of related RawFiles (e.g. all files in one service folder),
    generate structured artifact documents via OpenAI.

    Pass `artifact_types` to override auto-detection.
    """
    if not files:
        return []

    types_to_gen = artifact_types or _detect_relevant_types(files)
    artifacts: list[Artifact] = []

    for art_type in types_to_gen:
        log.info("Generating %s for %d files from %s …",
                 art_type, len(files), files[0].repo)
        user_msg = _build_user_message(files, art_type)

        try:
            resp = client.chat.completions.create(
                model=cfg.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            log.error("OpenAI error generating %s: %s", art_type, exc)
            continue

        # Extract a title from the first heading, fallback to type name
        title = art_type.replace("_", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        artifacts.append(Artifact(
            artifact_type=art_type,
            title=title,
            content=content,
            source_repo=files[0].repo,
            source_paths=[f.path for f in files],
            source_urls=[f.url  for f in files],
            language=files[0].language,
            metadata={
                "num_source_files": len(files),
                "branch": files[0].branch,
            },
        ))

    return artifacts


def group_files_by_directory(files: list[RawFile]) -> dict[str, list[RawFile]]:
    """
    Group files by their top-level directory (or repo root).
    Used to batch related files together before artifact generation.
    """
    groups: dict[str, list[RawFile]] = {}
    for f in files:
        parts = f.path.split("/")
        key = f"{f.repo}/{parts[0]}" if len(parts) > 1 else f"{f.repo}/__root__"
        groups.setdefault(key, []).append(f)
    return groups
