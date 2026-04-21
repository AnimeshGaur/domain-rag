"""
main.py
────────
FastAPI application entry point for RAGbase.

Run with:
    uvicorn main:app --reload --port 8000

API docs available at:
    http://localhost:8000/api/docs    (Swagger UI)
    http://localhost:8000/api/redoc  (ReDoc)
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="RAGbase — Docusaurus Intelligence",
    description=(
        "Phase 1: GitHub crawl → Docusaurus parse → Embed → Elasticsearch hybrid search + rerank. "
        "Phase 2: + LLM artifact generation and grounded answer synthesis."
    ),
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        cfg.FRONTEND_ORIGIN,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# ── Serve React build in production ───────────────────────────────────────────
frontend_build = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(frontend_build):
    app.mount("/", StaticFiles(directory=frontend_build, html=True), name="static")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    from search.elastic_store import ensure_index
    from embedding.base import get_embed_provider

    try:
        embed = get_embed_provider()
        ensure_index(embed_dims=embed.dims)
        log.info(
            "✓ Elasticsearch index ready | embed=%s(%s, %dd)",
            cfg.EMBED_PROVIDER, embed.model_name, embed.dims,
        )
    except Exception as exc:
        log.warning("Startup: could not connect to Elasticsearch or load embedding model: %s", exc)
        log.warning("The app will still start — run ingestion once ES is available.")

    if cfg.llm_enabled:
        log.info("✓ LLM enabled | provider=%s | model=%s", cfg.LLM_PROVIDER, cfg.OPENAI_CHAT_MODEL)
    else:
        log.info(
            "ℹ LLM disabled — /api/query will return 501. "
            "Set LLM_PROVIDER=openai + OPENAI_API_KEY to enable."
        )
