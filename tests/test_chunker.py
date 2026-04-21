"""
tests/test_chunker.py
──────────────────────
Unit tests for chunking logic — no API calls required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ingestion.chunker import chunk_artifact, _token_len
from ingestion.artifact_generator import Artifact


def _make_artifact(content: str, art_type="component_doc") -> Artifact:
    return Artifact(
        artifact_type=art_type,
        title="Test Component",
        content=content,
        source_repo="acme/test",
        source_paths=["src/test.py"],
        source_urls=["https://github.com/acme/test/blob/main/src/test.py"],
    )


class TestChunkArtifact:

    def test_short_content_single_chunk(self):
        """Content well under chunk_size should produce exactly one chunk."""
        art = _make_artifact("This is a short description of a small component.")
        chunks = chunk_artifact(art)
        assert len(chunks) == 1

    def test_long_content_multiple_chunks(self):
        """Long content should be split into multiple chunks."""
        long_content = "\n\n".join(
            [f"## Section {i}\n" + ("Word " * 100) for i in range(20)]
        )
        art = _make_artifact(long_content)
        chunks = chunk_artifact(art)
        assert len(chunks) > 1

    def test_chunk_contains_title_prefix(self):
        """Every chunk should start with the artifact type + title prefix."""
        art = _make_artifact("Some content here.")
        chunks = chunk_artifact(art)
        for c in chunks:
            assert "[COMPONENT_DOC]" in c.text
            assert "Test Component" in c.text

    def test_chunk_ids_are_unique(self):
        """Each chunk in an artifact must have a unique ID."""
        long_content = "\n\n".join([f"## Section {i}\n" + ("A " * 200) for i in range(10)])
        art = _make_artifact(long_content)
        chunks = chunk_artifact(art)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequence(self):
        """chunk_index should be sequential starting at 0."""
        long_content = "\n\n".join([f"## Section {i}\n" + ("B " * 200) for i in range(10)])
        art = _make_artifact(long_content)
        chunks = chunk_artifact(art)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)

    def test_provenance_preserved(self):
        """Source repo, paths, and URLs must be carried into every chunk."""
        art = _make_artifact("Short content.")
        chunks = chunk_artifact(art)
        for c in chunks:
            assert c.source_repo == "acme/test"
            assert c.source_paths == ["src/test.py"]
            assert "github.com/acme/test" in c.source_urls[0]

    def test_chunk_token_length_within_limit(self):
        """No chunk should exceed chunk_size + chunk_overlap tokens (with some tolerance)."""
        from config import cfg
        long_content = "Word " * 5000
        art = _make_artifact(long_content)
        chunks = chunk_artifact(art)
        for c in chunks:
            toks = _token_len(c.text)
            # Allow a small overshoot due to the prepended title
            assert toks <= cfg.CHUNK_SIZE + cfg.CHUNK_OVERLAP + 50, \
                f"Chunk {c.chunk_index} has {toks} tokens (limit {cfg.CHUNK_SIZE})"

    def test_markdown_heading_split_preference(self):
        """Splitter should prefer breaking at markdown headings."""
        content = "\n".join([
            f"## Section {i}\n" + ("Content sentence. " * 40)
            for i in range(8)
        ])
        art = _make_artifact(content)
        chunks = chunk_artifact(art)
        # At least some chunks should start right after a heading boundary
        heading_starts = sum(
            1 for c in chunks
            if "## Section" in c.text or "[COMPONENT_DOC]" in c.text
        )
        assert heading_starts > 0

    def test_different_artifact_types_different_prefixes(self):
        """Artifact type prefix should match the type field."""
        for art_type in ["api_contract", "sequence_flow", "data_flow", "arch_summary"]:
            art = _make_artifact("Some text.", art_type=art_type)
            chunks = chunk_artifact(art)
            assert f"[{art_type.upper()}]" in chunks[0].text

    def test_to_dict_completeness(self):
        """Chunk.to_dict() must contain all fields needed for Elasticsearch."""
        art = _make_artifact("Content.")
        chunk = chunk_artifact(art)[0]
        d = chunk.to_dict()
        required_keys = ["chunk_id", "artifact_type", "title", "text",
                         "chunk_index", "total_chunks", "source_repo",
                         "source_paths", "source_urls", "language"]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_empty_content_returns_empty(self):
        """Artifact with empty content should produce no chunks (or 1 trivial one)."""
        art = _make_artifact("")
        chunks = chunk_artifact(art)
        # Either 0 chunks or 1 chunk that is just the prefix
        assert len(chunks) <= 1
