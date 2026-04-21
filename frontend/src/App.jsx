import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000/api";

// ── Utilities ──────────────────────────────────────────────────────────────

const DOC_TYPE_COLOR = {
  api_ref:        "#378ADD",
  tutorial:       "#3BAD82",
  guide:          "#7F77DD",
  concept:        "#D85A30",
  business:       "#BA7517",
  blog:           "#888",
  page:           "#555",
  doc:            "#6B6882",
  image:          "#6B9",
  component_doc:  "#7F77DD",
  api_contract:   "#378ADD",
  sequence_flow:  "#D85A30",
  data_flow:      "#3BAD82",
  arch_summary:   "#BA7517",
  business_process:"#E07820",
  domain_model:   "#9C27B0",
  user_guide:     "#3BAD82",
};

const DOC_TYPE_LABEL = {
  api_ref: "API Ref", tutorial: "Tutorial", guide: "Guide",
  concept: "Concept", business: "Business", blog: "Blog", page: "Page",
  doc: "Doc", image: "Image",
  component_doc: "Component", api_contract: "API Contract",
  sequence_flow: "Sequence", data_flow: "Data Flow", arch_summary: "Architecture",
  business_process: "Business Process", domain_model: "Domain Model", user_guide: "User Guide",
};

function typeColor(t) { return DOC_TYPE_COLOR[t] || "#888"; }
function typeLabel(t) { return DOC_TYPE_LABEL[t] || t; }

function renderMd(text) {
  if (!text) return "";
  return text
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre class="code-block"><code class="lang-${lang}">${code.trim().replace(/</g,"&lt;").replace(/>/g,"&gt;")}</code></pre>`)
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,  "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[Source (\d+)\]/g, '<cite class="source-ref">[S$1]</cite>')
    .replace(/\n/g, "<br/>");
}

// ── Small reusable components ──────────────────────────────────────────────

function Badge({ type, label }) {
  const color = typeColor(type);
  return (
    <span className="badge" style={{ background: color + "22", color, border: `1px solid ${color}55` }}>
      {label || typeLabel(type)}
    </span>
  );
}

function ResultCard({ result, index }) {
  const [expanded, setExpanded] = useState(false);
  const color = typeColor(result.artifact_type || result.doc_type);
  return (
    <div className="result-card" style={{ borderLeftColor: color }}>
      <div className="result-header">
        <span className="result-index" style={{ background: color }}>#{index + 1}</span>
        <Badge type={result.artifact_type || result.doc_type} />
        {result.category && <span className="result-category">{result.category}</span>}
        <span className="result-score">{(result.rerank_score ?? result.score ?? 0).toFixed(3)}</span>
      </div>
      <div className="result-title">{result.title}</div>
      {result.heading_path && <div className="result-path">{result.heading_path}</div>}
      <div
        className={`result-text ${expanded ? "expanded" : ""}`}
        dangerouslySetInnerHTML={{ __html: renderMd(result.text || result.description || "") }}
      />
      <div className="result-footer">
        <a href={result.url} target="_blank" rel="noreferrer" className="result-url">
          {result.url?.replace("https://github.com/", "⌁ ")}
        </a>
        <button className="btn-expand" onClick={() => setExpanded(v => !v)}>
          {expanded ? "▲ Less" : "▼ More"}
        </button>
      </div>
      {result.image_refs?.length > 0 && (
        <div className="result-images">
          {result.image_refs.slice(0, 3).map((img, i) => (
            <span key={i} className="img-ref">🖼 {img.split("/").pop()}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function AnswerSource({ source, index }) {
  const color = typeColor(source.artifact_type || source.doc_type);
  return (
    <div className="source-card" style={{ borderLeftColor: color }}>
      <div className="source-header">
        <span className="source-index" style={{ background: color }}>S{index + 1}</span>
        <Badge type={source.artifact_type || source.doc_type} />
        {source.category && <span className="result-category">{source.category}</span>}
        <span className="result-score">{source.score?.toFixed(3)}</span>
      </div>
      <div className="result-title">{source.title}</div>
      <a href={source.url} target="_blank" rel="noreferrer" className="result-url">
        {source.url?.replace("https://github.com/", "⌁ ")}
      </a>
    </div>
  );
}

function IngestPanel({ onDone, llmEnabled }) {
  const [repos, setRepos] = useState("");
  const [useLlm, setUseLlm] = useState(false);
  const [reset, setReset] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);

  const pollStatus = useCallback(async () => {
    const r = await fetch(`${API}/status`);
    const d = await r.json();
    setStatus(d);
    if (!d.ingestion.running && polling) { setPolling(false); setLoading(false); onDone?.(); }
  }, [polling, onDone]);

  useEffect(() => { if (!polling) return; const id = setInterval(pollStatus, 3000); return () => clearInterval(id); }, [polling, pollStatus]);
  useEffect(() => { pollStatus(); }, []);

  async function startIngest() {
    setLoading(true);
    await fetch(`${API}/ingest`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repos: repos.split(",").map(s => s.trim()).filter(Boolean),
        full_reset: reset, use_llm: useLlm,
      }),
    });
    setPolling(true);
  }

  const running = status?.ingestion?.running;
  const last = status?.ingestion?.last_result;
  const docs = status?.es_index?.doc_count ?? "–";
  const cfg = status?.config || {};

  return (
    <div className="ingest-panel">
      <div className="panel-title">⚙ Ingestion Control</div>
      <div className="stat-row">
        <div className="stat"><span className="stat-val">{docs}</span><span className="stat-lbl">chunks indexed</span></div>
        <div className="stat"><span className={`stat-val ${running ? "pulse" : ""}`}>{running ? "Running" : "Idle"}</span><span className="stat-lbl">pipeline</span></div>
      </div>
      {last && !last.error && (
        <div className="last-run">
          Last: {last.docs_crawled}↓ → {last.chunks_indexed}✦ in {last.duration_seconds}s [{last.mode}]
          {last.errors?.length > 0 && <span className="warn"> {last.errors.length} warnings</span>}
        </div>
      )}

      <div className="config-pill-row">
        <span className="config-pill">embed: {cfg.embed_provider}/{cfg.embed_model?.split("/").pop()}</span>
        <span className={`config-pill ${cfg.llm_enabled ? "active" : "off"}`}>
          llm: {cfg.llm_enabled ? cfg.llm_model : "disabled"}
        </span>
      </div>

      <label className="field-label">GitHub repos (comma-separated)</label>
      <input className="text-input" placeholder="acme/backend-api, acme/docs" value={repos}
        onChange={e => setRepos(e.target.value)} disabled={loading} />

      <label className="toggle-row">
        <input type="checkbox" checked={reset} onChange={e => setReset(e.target.checked)} />
        <span>Full reset (wipe index first)</span>
      </label>

      {llmEnabled && (
        <label className="toggle-row">
          <input type="checkbox" checked={useLlm} onChange={e => setUseLlm(e.target.checked)} />
          <span>Phase 2: LLM artifact generation</span>
        </label>
      )}

      <button className={`btn-ingest ${loading ? "loading" : ""}`} onClick={startIngest} disabled={loading}>
        {loading ? "Pipeline running…" : `▶ Run ${useLlm ? "Phase 2 (LLM)" : "Phase 1"} Ingestion`}
      </button>
    </div>
  );
}

function FilterBar({ sources, onFilter, activeFilters }) {
  const [docType, setDocType] = useState("");
  const [category, setCategory] = useState("");

  function apply() {
    const f = {};
    if (docType) f.doc_type = docType;
    if (category) f.category = category;
    onFilter(Object.keys(f).length ? f : null);
  }
  function clear() { setDocType(""); setCategory(""); onFilter(null); }

  return (
    <div className="filter-bar">
      <select className="filter-select" value={docType} onChange={e => setDocType(e.target.value)}>
        <option value="">All types</option>
        {(sources?.doc_types || []).map(t => <option key={t} value={t}>{typeLabel(t)}</option>)}
      </select>
      <select className="filter-select" value={category} onChange={e => setCategory(e.target.value)}>
        <option value="">All categories</option>
        {(sources?.categories || []).map(c => <option key={c} value={c}>{c}</option>)}
      </select>
      <button className="btn-filter" onClick={apply}>Apply</button>
      {activeFilters && <button className="btn-clear-filter" onClick={clear}>✕</button>}
    </div>
  );
}

// ── Search Tab ────────────────────────────────────────────────────────────────

function SearchTab({ sources, llmEnabled }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState(null);
  const inputRef = useRef(null);

  async function doSearch() {
    if (!query.trim() || loading) return;
    setLoading(true);
    try {
      const r = await fetch(`${API}/search`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query, filters, top_k: 20, rerank_top_n: 8 }),
      });
      const d = await r.json();
      setResults(d);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  function handleKey(e) { if (e.key === "Enter") doSearch(); }

  const suggestions = [
    "What is the authentication flow?",
    "How do I configure the API gateway?",
    "What are the main business processes?",
    "List all REST API endpoints",
    "How does data flow through the system?",
  ];

  return (
    <div className="tab-content">
      <div className="search-bar">
        <input className="search-input" ref={inputRef} placeholder="Search your Docusaurus docs…"
          value={query} onChange={e => setQuery(e.target.value)} onKeyDown={handleKey} disabled={loading} />
        <button className={`btn-search ${loading ? "loading" : ""}`} onClick={doSearch} disabled={loading || !query.trim()}>
          {loading ? "…" : "⌕"}
        </button>
      </div>

      <FilterBar sources={sources} onFilter={setFilters} activeFilters={filters} />

      {filters && (
        <div className="active-filter">
          Filtering: {Object.entries(filters).map(([k,v]) => `${k}=${v}`).join(", ")}
        </div>
      )}

      {results === null && (
        <div className="empty-state">
          <div className="empty-icon">⌕</div>
          <h2>Search your documentation</h2>
          <p>Hybrid semantic + keyword search with cross-encoder reranking.<br/>No LLM required — works immediately after indexing.</p>
          <div className="suggestions">
            {suggestions.map((s, i) => (
              <button key={i} className="suggestion" onClick={() => { setQuery(s); inputRef.current?.focus(); }}>{s}</button>
            ))}
          </div>
        </div>
      )}

      {results && (
        <div className="results-area">
          <div className="results-meta">
            {results.retrieval_hits} candidates → {results.reranked_to} results for <em>{results.question}</em>
          </div>
          {results.results?.length === 0 && (
            <div className="no-results">No results found. Try different keywords or check if docs are indexed.</div>
          )}
          {results.results?.map((r, i) => <ResultCard key={r.chunk_id} result={r} index={i} />)}
        </div>
      )}
    </div>
  );
}

// ── Ask Tab (LLM) ─────────────────────────────────────────────────────────────

function AskTab({ sources, llmEnabled }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function sendMessage() {
    const q = input.trim();
    if (!q || loading) return;
    setInput(""); setLoading(true);
    const userMsg = { role: "user", content: q, id: Date.now() };
    const asstMsg = { role: "assistant", content: "", id: Date.now() + 1, streaming: true };
    setMessages(prev => [...prev, userMsg, asstMsg]);

    try {
      const resp = await fetch(`${API}/query/stream`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, filters }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        setMessages(prev => prev.map(m => m.id === asstMsg.id
          ? { ...m, content: `⚠ ${err.detail || "LLM not available. Switch to the Search tab."}`, streaming: false } : m));
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") continue;
          try {
            const parsed = JSON.parse(payload);
            const token = parsed.token || "";
            const metaMatch = token.match(/<!--RAG_META:(.+?)-->/);
            if (metaMatch) {
              const meta = JSON.parse(metaMatch[1]);
              setMessages(prev => prev.map(m => m.id === asstMsg.id
                ? { ...m, content: m.content.replace(/<!--RAG_META:.+?-->/, "").trim(),
                    sources: meta.__sources__, retrieval_hits: meta.__retrieval_hits__, streaming: false } : m));
            } else {
              setMessages(prev => prev.map(m => m.id === asstMsg.id ? { ...m, content: m.content + token } : m));
            }
          } catch {}
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m => m.id === asstMsg.id
        ? { ...m, content: `Error: ${err.message}`, streaming: false } : m));
    } finally {
      setLoading(false); setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
      inputRef.current?.focus();
    }
  }

  function handleKey(e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }

  const suggestions = [
    "Explain the authentication and authorization flow", "What API endpoints are available?",
    "Describe the main business processes", "How does data move through the system?",
    "What are the key architectural decisions?",
  ];

  return (
    <div className="tab-content ask-tab">
      {!llmEnabled && (
        <div className="llm-banner">
          ⚠ LLM is disabled. Set <code>LLM_PROVIDER</code> (e.g. <code>groq</code> or <code>openai</code>) 
          and your corresponding API Key in your .env to enable AI answers. 
          The <strong>Search</strong> tab works without LLM.
        </div>
      )}
      <FilterBar sources={sources} onFilter={setFilters} activeFilters={filters} />
      {filters && <div className="active-filter">Filtering: {Object.entries(filters).map(([k,v]) => `${k}=${v}`).join(", ")}</div>}

      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">◈</div>
          <h2>Ask anything about your codebase</h2>
          <p>LLM-grounded answers with citations from your Docusaurus documentation.</p>
          <div className="suggestions">
            {suggestions.map((s, i) => <button key={i} className="suggestion" onClick={() => { setInput(s); inputRef.current?.focus(); }}>{s}</button>)}
          </div>
        </div>
      ) : (
        <div className="messages">
          {messages.map(msg => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="msg-role">{msg.role === "user" ? "You" : "RAGbase"}</div>
              <div className="msg-body" dangerouslySetInnerHTML={{ __html: renderMd(msg.content) }} />
              {msg.sources?.length > 0 && (
                <div className="sources-section">
                  <div className="sources-label">Sources ({msg.sources.length})</div>
                  <div className="sources-grid">
                    {msg.sources.map((s, i) => <AnswerSource key={i} source={s} index={i} />)}
                  </div>
                </div>
              )}
              {msg.retrieval_hits != null && (
                <div className="msg-meta">{msg.retrieval_hits} chunks retrieved · {msg.sources?.length} after rerank</div>
              )}
            </div>
          ))}
          {messages.at(-1)?.streaming && <div className="typing-dot" />}
          <div ref={bottomRef} />
        </div>
      )}

      <div className="input-area">
        <div className="input-row">
          <textarea ref={inputRef} className="chat-input" rows={2}
            placeholder="Ask a question about your documentation…  (Enter to send)"
            value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey} disabled={loading} />
          <button className={`btn-send ${loading ? "loading" : ""}`} onClick={sendMessage}
            disabled={loading || !input.trim()}>{loading ? "…" : "↑"}</button>
        </div>
        <div className="input-hint">
          Hybrid kNN + BM25 · Cross-encoder rerank · {llmEnabled ? "GPT-4o grounded answer" : "Search-only mode (LLM disabled)"}
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState("search");
  const [sidebarOpen, setSidebar] = useState(true);
  const [sources, setSources] = useState(null);
  const [llmEnabled, setLlmEnabled] = useState(false);

  useEffect(() => {
    fetch(`${API}/status`).then(r => r.json()).then(d => setLlmEnabled(!!d.config?.llm_enabled)).catch(() => {});
    fetch(`${API}/sources`).then(r => r.json()).then(setSources).catch(() => {});
  }, []);

  return (
    <div className="app" data-sidebar={sidebarOpen}>
      <style>{STYLES}</style>

      <header className="header">
        <div className="header-left">
          <button className="sidebar-toggle" onClick={() => setSidebar(v => !v)}>{sidebarOpen ? "◀" : "▶"}</button>
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">RAG<em>base</em></span>
          </div>
          <div className="header-sub">Docusaurus Intelligence</div>
        </div>
        <div className="tab-switcher">
          <button className={`tab-btn ${tab === "search" ? "active" : ""}`} onClick={() => setTab("search")}>
            ⌕ Search
          </button>
          <button className={`tab-btn ${tab === "ask" ? "active" : ""} ${!llmEnabled ? "dim" : ""}`} onClick={() => setTab("ask")}>
            ◈ Ask {!llmEnabled && <span className="tab-badge">LLM off</span>}
          </button>
        </div>
      </header>

      <div className="layout">
        {sidebarOpen && (
          <aside className="sidebar">
            <IngestPanel onDone={() => fetch(`${API}/sources`).then(r=>r.json()).then(setSources).catch(()=>{})} llmEnabled={llmEnabled} />
          </aside>
        )}
        <main className="main-area">
          {tab === "search" ? (
            <SearchTab sources={sources} llmEnabled={llmEnabled} />
          ) : (
            <AskTab sources={sources} llmEnabled={llmEnabled} />
          )}
        </main>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0D0D10; --bg2: #13131A; --bg3: #1A1A24;
    --border: #252535; --text: #E8E6F0; --muted: #6B6882;
    --accent: #7F77DD; --accent2: #3BAD82; --user-bg: #1C1C2E; --asst-bg: #13131A;
    --radius: 12px; --font: 'Sora', sans-serif; --mono: 'JetBrains Mono', monospace;
    --hdr: 52px; --sidebar-w: 320px;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--font); }
  .app { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* ── Header ── */
  .header {
    height: var(--hdr); background: var(--bg2); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 20px; gap: 16px; flex-shrink: 0; z-index: 10; position: relative;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }

  .sidebar-toggle {
    background: none; border: 1px solid var(--border); color: var(--muted);
    padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px;
    transition: border-color .2s, color .2s;
  }
  .sidebar-toggle:hover { border-color: var(--accent); color: var(--accent); }

  .logo { display: flex; align-items: center; gap: 8px; }
  .logo-icon { font-size: 20px; color: var(--accent); }
  .logo-text { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
  .logo-text em { color: var(--accent); font-style: normal; }
  .header-sub { font-size: 11px; color: var(--muted); }

  /* ── Tab switcher ── */
  .tab-switcher { display: flex; gap: 4px; }
  .tab-btn {
    padding: 6px 18px; border-radius: 8px; border: 1px solid var(--border);
    background: none; color: var(--muted); font-family: var(--font); font-size: 13px;
    cursor: pointer; transition: all .2s; display: flex; align-items: center; gap: 6px;
  }
  .tab-btn:hover { border-color: var(--accent); color: var(--text); }
  .tab-btn.active { background: var(--accent); color: #fff; border-color: transparent; }
  .tab-btn.dim { opacity: .6; }
  .tab-badge { font-size: 9px; padding: 1px 5px; background: #333; border-radius: 4px; }

  /* ── Layout ── */
  .layout { display: flex; flex: 1; overflow: hidden; }
  .sidebar {
    width: var(--sidebar-w); flex-shrink: 0;
    background: var(--bg2); border-right: 1px solid var(--border);
    overflow-y: auto; padding: 20px;
  }
  .main-area { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .tab-content { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }

  /* ── Ingest panel ── */
  .ingest-panel { display: flex; flex-direction: column; gap: 12px; }
  .panel-title { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; color: var(--muted); text-transform: uppercase; }
  .stat-row { display: flex; gap: 10px; }
  .stat { flex: 1; background: var(--bg3); border-radius: var(--radius); padding: 12px; display: flex; flex-direction: column; align-items: center; }
  .stat-val { font-size: 20px; font-weight: 700; font-family: var(--mono); color: var(--accent); }
  .stat-val.pulse { animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .stat-lbl { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .last-run { font-size: 10px; color: var(--muted); line-height: 1.5; }
  .warn { color: #BA7517; margin-left: 4px; }
  .config-pill-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .config-pill {
    font-size: 9px; padding: 2px 8px; border-radius: 10px;
    background: var(--bg3); border: 1px solid var(--border); color: var(--muted); font-family: var(--mono);
  }
  .config-pill.active { border-color: var(--accent2); color: var(--accent2); }
  .config-pill.off { border-color: #444; color: #555; }
  .field-label { font-size: 10px; color: var(--muted); font-weight: 600; letter-spacing: .5px; }
  .text-input {
    width: 100%; padding: 8px 12px; border-radius: 8px;
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text); font-size: 12px; font-family: var(--mono); transition: border-color .2s;
  }
  .text-input:focus { outline: none; border-color: var(--accent); }
  .toggle-row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .toggle-row input { accent-color: var(--accent); }
  .btn-ingest {
    padding: 10px; border-radius: 8px; border: none; cursor: pointer;
    background: var(--accent); color: #fff; font-family: var(--font);
    font-size: 13px; font-weight: 600; transition: opacity .2s, transform .1s;
  }
  .btn-ingest:hover:not(:disabled) { opacity: .88; transform: translateY(-1px); }
  .btn-ingest.loading { opacity: .6; cursor: not-allowed; }

  /* ── Search bar ── */
  .search-bar { display: flex; gap: 10px; align-items: center; }
  .search-input {
    flex: 1; padding: 12px 18px; border-radius: 10px; font-size: 15px;
    background: var(--bg2); border: 1px solid var(--border);
    color: var(--text); font-family: var(--font); transition: border-color .2s;
  }
  .search-input:focus { outline: none; border-color: var(--accent); }
  .btn-search {
    width: 48px; height: 48px; border-radius: 10px; border: none; cursor: pointer;
    background: var(--accent); color: #fff; font-size: 22px;
    transition: opacity .2s, transform .1s; flex-shrink: 0;
  }
  .btn-search:hover:not(:disabled) { opacity: .85; transform: translateY(-1px); }
  .btn-search:disabled { opacity: .4; cursor: not-allowed; }

  /* ── Filters ── */
  .filter-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .filter-select {
    padding: 5px 10px; border-radius: 6px; background: var(--bg3); border: 1px solid var(--border);
    color: var(--text); font-size: 12px; cursor: pointer;
  }
  .btn-filter {
    padding: 5px 12px; border-radius: 6px; border: 1px solid var(--accent);
    background: none; color: var(--accent); font-size: 12px; cursor: pointer; transition: background .2s;
  }
  .btn-filter:hover { background: var(--accent); color: #fff; }
  .btn-clear-filter {
    padding: 4px 8px; border-radius: 6px; border: 1px solid var(--border);
    background: none; color: var(--muted); font-size: 11px; cursor: pointer;
  }
  .active-filter { font-size: 10px; color: var(--accent); font-family: var(--mono); }

  /* ── Result cards ── */
  .results-meta { font-size: 12px; color: var(--muted); }
  .results-area { display: flex; flex-direction: column; gap: 12px; }
  .no-results { color: var(--muted); text-align: center; padding: 40px; }

  .result-card {
    background: var(--bg2); border-radius: var(--radius); padding: 14px 16px;
    border-left: 3px solid var(--accent); display: flex; flex-direction: column; gap: 6px;
    transition: border-color .2s;
  }
  .result-card:hover { border-left-width: 4px; }
  .result-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .result-index { font-family: var(--mono); font-size: 10px; padding: 2px 7px; border-radius: 4px; color: #fff; }
  .result-score { font-size: 10px; color: var(--muted); font-family: var(--mono); margin-left: auto; }
  .result-title { font-size: 14px; font-weight: 600; color: var(--text); }
  .result-path { font-size: 10px; color: var(--muted); font-family: var(--mono); }
  .result-category { font-size: 10px; color: var(--muted); background: var(--bg3); padding: 1px 7px; border-radius: 10px; }
  .result-text { font-size: 13px; color: var(--muted); line-height: 1.6; max-height: 80px; overflow: hidden; transition: max-height .3s; }
  .result-text.expanded { max-height: 600px; }
  .result-text h1,.result-text h2,.result-text h3 { color: var(--accent); margin: 8px 0 4px; }
  .result-text code { font-family: var(--mono); font-size: 11px; background: #1E1E2E; padding: 1px 5px; border-radius: 3px; }
  .result-text .code-block { background: #0A0A12; border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin: 8px 0; overflow-x: auto; }
  .result-footer { display: flex; align-items: center; justify-content: space-between; }
  .result-url { font-size: 10px; color: var(--muted); font-family: var(--mono); word-break: break-all; }
  .result-url:hover { color: var(--accent); }
  .btn-expand { font-size: 10px; color: var(--muted); background: none; border: none; cursor: pointer; padding: 2px 8px; }
  .result-images { display: flex; gap: 8px; flex-wrap: wrap; }
  .img-ref { font-size: 10px; color: var(--accent2); background: var(--bg3); padding: 2px 8px; border-radius: 10px; font-family: var(--mono); }

  /* ── Badge ── */
  .badge { font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px; }

  /* ── Chat (Ask tab) ── */
  .ask-tab { padding: 0 !important; }
  .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 24px; }
  .message { display: flex; flex-direction: column; gap: 8px; max-width: 860px; }
  .message.user { align-self: flex-end; align-items: flex-end; }
  .message.assistant { align-self: flex-start; }
  .msg-role { font-size: 10px; font-weight: 600; letter-spacing: 1px; color: var(--muted); text-transform: uppercase; }
  .msg-body { padding: 16px 20px; border-radius: var(--radius); line-height: 1.7; font-size: 14px; }
  .message.user .msg-body { background: var(--user-bg); border: 1px solid #2A2A40; }
  .message.assistant .msg-body { background: var(--asst-bg); border: 1px solid var(--border); }
  .msg-body h1,.msg-body h2,.msg-body h3 { color: var(--accent); margin: 12px 0 6px; }
  .msg-body code { font-family: var(--mono); font-size: 12px; background: #1E1E2E; padding: 2px 6px; border-radius: 4px; }
  .msg-body .code-block { background: #0A0A12; border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 10px 0; overflow-x: auto; }
  .msg-body strong { color: #fff; }
  .source-ref { background: var(--accent); color: #fff; font-size: 10px; padding: 1px 5px; border-radius: 3px; font-family: var(--mono); vertical-align: super; }
  .msg-meta { font-size: 10px; color: var(--muted); }
  .sources-section { margin-top: 6px; }
  .sources-label { font-size: 10px; font-weight: 600; letter-spacing: 1px; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
  .sources-grid { display: flex; flex-direction: column; gap: 6px; }
  .source-card { background: var(--bg3); border-radius: 8px; padding: 8px 12px; border-left: 3px solid var(--accent); display: flex; flex-direction: column; gap: 4px; }
  .source-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .source-index { font-family: var(--mono); font-size: 10px; padding: 2px 6px; border-radius: 4px; color: #fff; }
  .typing-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); animation: blink 1s ease-in-out infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  /* ── Input area ── */
  .input-area { padding: 14px 24px 18px; border-top: 1px solid var(--border); background: var(--bg2); flex-shrink: 0; }
  .input-row { display: flex; gap: 10px; align-items: flex-end; }
  .chat-input {
    flex: 1; padding: 12px 16px; border-radius: 10px; resize: none;
    background: var(--bg3); border: 1px solid var(--border); color: var(--text);
    font-size: 14px; font-family: var(--font); line-height: 1.5; transition: border-color .2s;
  }
  .chat-input:focus { outline: none; border-color: var(--accent); }
  .btn-send {
    width: 44px; height: 44px; border-radius: 10px; border: none;
    background: var(--accent); color: #fff; font-size: 20px; cursor: pointer;
    transition: opacity .2s, transform .1s; flex-shrink: 0;
  }
  .btn-send:hover:not(:disabled) { opacity: .85; transform: translateY(-1px); }
  .btn-send:disabled { opacity: .4; cursor: not-allowed; }
  .btn-send.loading { animation: pulse 1s ease-in-out infinite; }
  .input-hint { font-size: 10px; color: var(--muted); margin-top: 6px; text-align: center; }

  /* ── LLM banner ── */
  .llm-banner {
    background: #1A1200; border: 1px solid #BA7517; border-radius: 8px;
    padding: 10px 16px; font-size: 12px; color: #E0A030; line-height: 1.6;
    flex-shrink: 0; margin: 0 24px;
  }
  .llm-banner code { font-family: var(--mono); background: #2A1E00; padding: 1px 5px; border-radius: 3px; }

  /* ── Empty state ── */
  .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; text-align: center; gap: 16px; }
  .empty-icon { font-size: 48px; color: var(--accent); opacity: .4; }
  .empty-state h2 { font-size: 22px; font-weight: 700; }
  .empty-state p { color: var(--muted); max-width: 480px; line-height: 1.7; }
  .suggestions { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; width: 100%; max-width: 540px; }
  .suggestion {
    padding: 11px 18px; border-radius: 10px; border: 1px solid var(--border);
    background: var(--bg2); color: var(--text); font-size: 13px; cursor: pointer;
    text-align: left; transition: border-color .2s, background .2s;
  }
  .suggestion:hover { border-color: var(--accent); background: var(--bg3); }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
`;
