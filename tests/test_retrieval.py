"""
tests/test_retrieval.py
────────────────────────
Integration tests for the retrieval layer.
Requires a running Elasticsearch instance and valid env vars.

Run with:
    pytest tests/test_retrieval.py -v --integration

Skip automatically if ES is unavailable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from dotenv import load_dotenv
load_dotenv()


def es_available() -> bool:
    try:
        from ingestion.elastic_store import es
        return es.ping()
    except Exception:
        return False


skip_no_es = pytest.mark.skipif(
    not es_available(),
    reason="Elasticsearch not available"
)


@skip_no_es
class TestHybridSearch:

    @pytest.fixture(autouse=True)
    def setup_test_index(self):
        """Create a temporary test index with a handful of documents."""
        import uuid
        from config import cfg
        from ingestion.elastic_store import es, ensure_index, upsert_chunks
        from ingestion.chunker import Chunk

        # Temporarily override index name
        original_index = cfg.ES_INDEX
        cfg.ES_INDEX = f"rag_test_{uuid.uuid4().hex[:8]}"
        ensure_index()

        # Create mock chunks + zero vectors (just testing the query flow)
        zero_vec = [0.0] * cfg.OPENAI_EMBED_DIMS
        test_chunks = [
            (Chunk(
                chunk_id=f"test_{i}",
                artifact_type="component_doc",
                title=f"AuthService" if i < 3 else "PaymentGateway",
                text=f"[COMPONENT_DOC] AuthService\n\nHandles user authentication via JWT tokens. Endpoint: POST /auth/login" if i < 3
                     else "[COMPONENT_DOC] PaymentGateway\n\nProcesses Stripe payments. Endpoint: POST /payment/charge",
                chunk_index=i,
                total_chunks=5,
                source_repo="acme/backend",
                source_paths=[f"src/auth.py"],
                source_urls=["https://github.com/acme/backend/blob/main/src/auth.py"],
                language="Python",
            ), zero_vec)
            for i in range(5)
        ]
        upsert_chunks(test_chunks)

        # Allow ES to index
        es.indices.refresh(index=cfg.ES_INDEX)

        yield

        # Teardown
        es.indices.delete(index=cfg.ES_INDEX)
        cfg.ES_INDEX = original_index

    def test_bm25_text_match(self):
        """BM25 search should return results for keyword matches."""
        from ingestion.elastic_store import hybrid_search
        from config import cfg
        zero_vec = [0.0] * cfg.OPENAI_EMBED_DIMS

        results = hybrid_search(
            query_vec=zero_vec,
            query_text="JWT authentication login",
            top_k=10,
        )
        assert len(results) > 0
        # Auth-related chunks should rank higher
        assert any("Auth" in r["title"] for r in results[:3])

    def test_metadata_filter(self):
        """Metadata filter should restrict results to matching artifact_type."""
        from ingestion.elastic_store import hybrid_search
        from config import cfg
        zero_vec = [0.0] * cfg.OPENAI_EMBED_DIMS

        results = hybrid_search(
            query_vec=zero_vec,
            query_text="authentication",
            top_k=10,
            filters={"artifact_type": "api_contract"},  # no api_contract docs in test set
        )
        assert len(results) == 0

    def test_returns_source_fields(self):
        """Each result must have the required fields for citation."""
        from ingestion.elastic_store import hybrid_search
        from config import cfg
        zero_vec = [0.0] * cfg.OPENAI_EMBED_DIMS

        results = hybrid_search(zero_vec, "payment", top_k=5)
        required = ["title", "artifact_type", "source_repo", "text"]
        for r in results:
            for field in required:
                assert field in r, f"Missing field '{field}' in result"


class TestReranker:

    def test_rerank_orders_by_relevance(self):
        """Re-ranker should put the more relevant chunk first."""
        from retrieval.reranker import rerank

        hits = [
            {"text": "The sky is blue and clouds are white.", "title": "Weather"},
            {"text": "JWT tokens are used for stateless authentication in REST APIs.", "title": "Auth"},
            {"text": "Python is a general purpose programming language.", "title": "Python"},
        ]
        ranked = rerank("How does authentication work with JWT?", hits, top_n=3)
        # Auth chunk should be #1
        assert ranked[0]["title"] == "Auth"

    def test_rerank_respects_top_n(self):
        """Re-ranker should return exactly top_n results."""
        from retrieval.reranker import rerank

        hits = [{"text": f"Doc {i}", "title": f"T{i}"} for i in range(10)]
        ranked = rerank("some query", hits, top_n=3)
        assert len(ranked) == 3

    def test_rerank_adds_score_field(self):
        """Each returned hit must have a 'rerank_score' field."""
        from retrieval.reranker import rerank

        hits = [{"text": "Some text about authentication.", "title": "Auth"}]
        ranked = rerank("authentication", hits, top_n=1)
        assert "rerank_score" in ranked[0]
        assert isinstance(ranked[0]["rerank_score"], float)

    def test_rerank_empty_input(self):
        """Re-ranker should handle empty hit list gracefully."""
        from retrieval.reranker import rerank
        result = rerank("query", [], top_n=5)
        assert result == []
