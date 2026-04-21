# RAGbase — Modular Docusaurus Intelligence

**RAGbase** is a production-grade, modular RAG (Retrieval-Augmented Generation) engine designed specifically to ingest GitHub repositories containing Docusaurus/Markdown documentation, and make it instantly searchable through a beautiful React interface.

It features a heavily optimized pipeline with **semantic chunking**, **BGE-backed vector search**, **manual Reciprocal Rank Fusion (RRF)**, and **cross-encoder reranking**. No paid Elasticsearch license is required! 

---

## ⚡️ Two-Phase Architecture

RAGbase is engineered to run in two distinct modes depending on your infrastructure and budget:

### Phase 1: Hybrid Search-Only (Free & Local)
Works entirely out-of-the-box using local Hugging Face models and the free tier of Elasticsearch. 
* **Ingestion:** Parses Markdown files, generating header-aware chunks, preserving fenced code blocks.
* **Embedding:** Uses local `BAAI/bge-small-en-v1.5` embeddings (384d, MTEB: 62.2).
* **Retrieval:** Runs separate kNN (semantic) and BM25 (keyword) queries.
* **Merging:** Performs manual Python-side Reciprocal Rank Fusion `score = Σ 1/(60 + rank)` to combine dense and keyword hits.
* **Reranking:** Passes the fused top-20 candidates through a local `ms-marco-MiniLM-L-6-v2` cross-encoder to produce the final top-8 results. 

### Phase 2: LLM "Ask" Mode (Optional)
Unlocks the conversational frontend UI (Ask Tab), enabling LLM-grounded question answering with precise citations (`[Source]`).
* Supports **OpenAI** (`gpt-4o`)
* Supports **Groq** (`qwen/qwen3-32b`)

---

## 🏗️ Project Layout

```
ragbase/
├── config.py                    # Central config handling multiple profiles (.env)
├── main.py                      # FastAPI app entry point (uvicorn)
├── requirements.txt
├── Dockerfile                   
├── docker-compose.yml           # Runs local Elasticsearch
│
├── api/                         
│   └── routes.py                # FastAPI endpoints (/ingest, /search, /query/stream)
│
├── crawling/
│   ├── github_crawler.py        # Robust fetcher (handles standard URLs & deep links)
│   └── docusaurus_parser.py     # Strips MDX & parses frontmatter
│
├── processing/
│   ├── chunker.py               # (ENHANCED) Header-aware semantic splitting
│   └── doc_schema.py            # Dataclasses (RawFile, ProcessedDoc, Chunk)
│
├── embedding/
│   ├── base.py                  # EmbedProvider Protocol 
│   ├── local_embed.py           # Local BGE embeddings with asymmetric query prefixes
│   └── openai_embed.py          # OpenAI embeddings (text-embedding-3-large)
│
├── search/
│   ├── elastic_store.py         # ES index management 
│   ├── hybrid_search.py         # Manual Python-side RRF (kNN + BM25)
│   └── reranker.py              # Cross-encoder pipeline (lazy loaded)
│
├── llm/
│   ├── base.py                  # LLMProvider Protocol
│   ├── openai_llm.py            # GPT-4o 
│   ├── groq_llm.py              # Groq (Qwen3-32B or customized via config)
│   └── answer_engine.py         # Prompts constraints, citations, formatting
│
└── frontend/                    
    └── src/App.jsx              # Beautiful React UI (Vite)
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Node.js 18+
- Docker + Docker Compose (for Elasticsearch)

### 2. Configure Environment
```bash
cp .env.example .env
```

Edit your `.env` file to customize your providers:
```ini
# Core Configuration
GITHUB_REPOS=owner/repo1, owner/repo2
ES_URL=http://localhost:9200
FRONTEND_ORIGIN=http://localhost:3000

# EMBED_PROVIDER can be 'local' (Free BGE) or 'openai'
EMBED_PROVIDER=local

# Phase 2 (Optional LLM) - Use 'openai', 'groq', or 'none'
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_key_here
```

### 3. Start Infrastructure & Backend
Start Elasticsearch (Docker) and the Python API.
```bash
# 1. Boot Elasticsearch
docker-compose up -d

# 2. Setup Python Virtual Env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the FastAPI application
uvicorn main:app --reload --port 8000
```

### 4. Start the Frontend
In a separate terminal, launch the React interface:
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### 5. Ingestion
You can ingest repositories directly from the React UI using the **⚙ Ingestion Control** sidebar, or run the CLI:
```bash
# Set HF_HOME if using local models to customize weight download location
export HF_HOME=./.cache
python3 -m cli.ingest --no-llm --repos your-org/docs-repo --full-reset
```

---

## 🛠️ Key Technical Features

### 1. Header-Aware Semantic Chunking 
Instead of flat `RecursiveCharacterTextSplitter`, RAGbase's `chunker.py` uses a **two-phase splitting approach**. It first divides documents precisely at heading boundaries (`# `, `## `, etc.), ensuring chunks never cross semantic sections. Code blocks are kept strictly atomic, preventing mid-code fragmentation. Short/empty chunks are automatically filtered.

### 2. Provider-Agnostic Embeddings 
Defaults to `BAAI/bge-small-en-v1.5` over `all-MiniLM-L6-v2` for vastly superior technical/code performance. RAGbase implements **Asymmetric Query Encoding** for BGE natively, ensuring that search queries map specifically to the passage vector space.

### 3. Manual Reciprocal Rank Fusion
Since Elasticsearch's free tier lacks native `.rrf`, RAGbase executes kNN (semantic) and BM25 (keyword) as two decoupled asynchronous queries and calculates the mathematical formula `Σ 1 / (60 + rank)` natively in Python. This achieves identical result relevance without the Enterprise license cost!

---

## 🔮 Roadmap / Proposed Improvements

1. **HyDE (Hypothetical Document Embeddings):** Embed an LLM-generated "ideal answer" to improve vector proximity.
2. **Parent-Child Chunking:** Index micro-chunks for high search recall, but pass the full parent document context to the LLM.
3. **ColBERT Late Interaction:** High-precision token-by-token comparison vs flat vector space.
4. **Git Cache / Incremental Ingestion:** Track Git Blob SHAs to prevent re-embedding identical files.
