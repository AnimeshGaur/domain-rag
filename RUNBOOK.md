# RAGbase Runbook

This guide covers the day-to-day operations of RAGbase—how to index new repositories, how to run queries from the terminal, and how to use the beautiful React frontend.

---

## 1. Starting the Infrastructure

Before running any CLI queries or using the UI, ensure your backend infrastructure is running:

### Step 1: Start Elasticsearch
Ensure Docker is running on your machine, then boot the detached Elasticsearch container:
```bash
docker compose up -d elasticsearch
```

### Step 2: Start the FastAPI Backend
You need the Python backend running for both the CLI query endpoints and the Frontend UI to communicate with Elasticsearch and the LLM.

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Optional: Set HF_HOME to avoid re-downloading local embed models into random directories
export HF_HOME=./.cache

# Start the API on port 8000
uvicorn main:app --reload --port 8000
```

---

## 2. Using the User Interface (UI)

The UI is the easiest way to interact with your vector-search engine.

### Launching the Frontend
Open a **new terminal window**:
```bash
cd frontend
npm install   # Only needed the very first time
npm run dev
```

Then, open **http://localhost:3000** in your web browser.

### Ingesting via the UI
1. Click the **⚙ (Gear Icon)** in the top right of the UI to open the Ingestion Panel.
2. Type in a GitHub repository (e.g. `owner/repo`) and hit **Enter** to add it to the list.
3. Check **"Full Reset"** if you want to wipe the Elasticsearch index clean, else leave it unchecked.
4. Click **Run Ingestion**. The progress spinner will show you the exact status of the backend crawling and embedding process without freezing the UI.

### Searching via the UI
1. Ensure your `.env` is configured correctly.
   - If `LLM_PROVIDER=openai` or `groq`, the **Ask Tab** will be unlocked.
   - If `LLM_PROVIDER=none`, only the **Search Tab** will work (hybrid search only).
2. Type your question. The system will automatically use **HyDE** to enhance your query and **Multi-Query Expansion** to find the absolute best hit. The LLM response will stream back in real-time.

---

## 3. Using the CLI

If you prefer operating from the terminal, or want to trigger ingestion via cron-jobs, use the CLI scripts. Ensure your `.venv` is activated.

### Ingesting via the CLI

The ingestion script automatically crawls, chunks, and embeds your documentation.

```bash
# Standard Ingestion (adds repos to your existing index)
python3 -m cli.ingest --no-llm --repos your-org/repo1 your-org/repo2

# Wipe the database and start fresh
python3 -m cli.ingest --no-llm --repos your-org/repo1 --full-reset
```

> **Note:** The `--no-llm` flag prevents the ingestion pipeline from calling the LLM to generate architecture summaries for raw code. If you want the LLM to summarize raw code before embedding, omit the flag (but ensure `LLM_PROVIDER` and API keys are set in your `.env`).

### Querying via the CLI

Test search relevance and LLM answering straight from the console. 
The `--search` command triggers the entire retrieval pipeline:
1. HyDE generation
2. Multi-Query Expansion
3. Manual kNN + BM25 RRF
4. Parent Chunk context expansion
5. LLM Synthesis

```bash
# Simple interactive query
python3 -m cli.query "How do I start the web worker?"

# Filter queries to a specific repo
python3 -m cli.query --search "deploy settings" --filter repo=code-chaser/hospital-management-system
```
