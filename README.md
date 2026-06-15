# RAG Document Assistant

> A full-stack, locally-running **Retrieval-Augmented Generation (RAG)** system — ask questions about your own PDF documents using hybrid search, cross-encoder reranking, and a local LLM via Ollama. Runs 100% offline with no external API keys required.

---

## Architecture Overview

```
┌─────────────────────────────┐        REST / Streaming HTTP
│  React Frontend  (port 5173) │ ◄──────────────────────────────► ┌──────────────────────────────┐
│  Vite · TailwindCSS          │                                   │  FastAPI Backend  (port 8000) │
│  React 18 + lucide-react     │                                   │  uvicorn · python-multipart   │
└─────────────────────────────┘                                   └──────────────┬───────────────┘
                                                                                 │
                                                          ┌──────────────────────▼───────────────────────┐
                                                          │             RAG Pipeline                      │
                                                          │  PDF Loader → Chunker → Embeddings            │
                                                          │  Chroma VectorStore  +  BM25 Index            │
                                                          │  Hybrid Search → CrossEncoder Rerank          │
                                                          │  Ollama LLM (phi3) → Streaming Tokens         │
                                                          └───────────────────────────────────────────────┘
```

---

## Features

- **Hybrid Search** — Combines dense vector similarity (Chroma + `all-MiniLM-L6-v2`) with sparse keyword matching (BM25Okapi) for maximum recall.
- **Cross-Encoder Reranking** — Retrieved candidates are rescored by `ms-marco-MiniLM-L-6-v2` for precision.
- **Streaming LLM Responses** — Token-by-token streaming via Ollama (`phi3`) with a live blinking cursor in the UI.
- **Source Attribution** — Every answer includes cited document & page references parsed from the `|||SOURCES|||` stream trailer.
- **Conversational Memory** — Rolling multi-turn chat history (last 5 turns) sent with every query for context.
- **Persistent Vectorstore** — Chroma DB is persisted to disk; previous documents survive server restarts.
- **Bootstrap on Startup** — Server automatically reloads the Chroma DB and rebuilds the BM25 index from disk on every restart.
- **Deduplication Guard** — Re-uploading the same filename is silently skipped, preventing duplicate chunks.
- **Drag-and-Drop Upload** — Browser-native PDF drag-and-drop with real-time progress indicator (uploading → embedding → done).
- **System Status Polling** — Frontend polls `/api/status` every 15 seconds for live Ollama health, chunk counts, and model info.
- **Stop Generation** — Users can abort a streaming response mid-flight via the Stop button.
- **Clear Controls** — Clear chat history or wipe the entire vectorstore from the sidebar.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 + | Python 3.14 works with warnings |
| Node.js | 18 + | For the React/Vite frontend |
| Ollama | Latest | Must be running locally |
| `phi3` model | — | `ollama pull phi3` |

---

## Quickstart

### 1 — Start Ollama

```powershell
ollama serve
```

Pull the model if you haven't already:

```powershell
ollama pull phi3
```

---

### 2 — Start the FastAPI Backend

```powershell
cd backend
pip install -r requirements.txt
python run.py
```

API will be live at **http://localhost:8000**
Swagger docs at **http://localhost:8000/docs**

---

### 3 — Start the React Frontend

```powershell
cd frontend
npm install
npm run dev
```

App will be live at **http://localhost:5173**

> The Vite dev server automatically proxies all `/api/*` requests to the FastAPI backend on port 8000 — no CORS issues during development.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health — Ollama state, active model, chunk count, config |
| `GET` | `/api/documents` | List all ingested document filenames |
| `POST` | `/api/upload` | Upload one or more PDF files for ingestion |
| `POST` | `/api/chat` | Send a query + history; streams plain-text tokens |
| `DELETE` | `/api/documents` | Wipe the entire vectorstore and reset pipeline state |
| `GET` | `/docs` | Auto-generated Swagger UI (FastAPI built-in) |

### Chat Request Schema

```json
POST /api/chat
{
  "query": "What is this document about?",
  "history": [
    ["user", "previous question"],
    ["assistant", "previous answer"]
  ]
}
```

### Upload Response Schema

```json
{
  "loaded":       ["file1.pdf"],
  "skipped":      [],
  "failed":       [],
  "chunks":       42,
  "total_chunks": 182
}
```

### Chat Response

Plain-text streaming response. Sources are appended as a trailer at the very end of the stream:

```
...last token of the answer|||SOURCES|||filename.pdf — Page 3|||filename.pdf — Page 7
```

---

## Configuration

Copy `.env.example` to `.env` inside the `backend/` directory and adjust as needed:

```powershell
cd backend
copy .env.example .env
```

```env
# Model settings
MODEL_NAME=phi3
EMBED_MODEL=all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
OLLAMA_BASE=http://localhost:11434

# RAG hyper-parameters
K_RETRIEVE=4
CHUNK_SIZE=300
CHUNK_OVERLAP=50
MAX_HISTORY_TURNS=5

# CORS (add frontend origins as comma-separated list)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Paths (relative to backend/)
VECTORSTORE_DIR=vectorstore
DOCUMENTS_DIR=documents
```

---

## Project Structure

```
Rag-master/
│
├── README.md
├── claude_info.md                # Project Q&A / info sheet
├── documents/                    # Drop PDFs here for bulk ingestion (CLI)
│   └── .gitkeep
│
├── vectorstore/                  # Persisted Chroma DB (auto-created at runtime)
│
├── backend/                      # FastAPI production backend
│   ├── run.py                    # Dev server launcher — uvicorn on port 8000
│   ├── ingest.py                 # Standalone CLI ingestion script (batch PDFs)
│   ├── evaluate.py               # RAGAS evaluation harness (Faithfulness, Relevancy, Precision)
│   ├── requirements.txt          # All Python dependencies
│   ├── .env.example              # Environment variable template
│   ├── .env                      # Your local config (not committed)
│   ├── documents/                # Backend-side PDF storage
│   ├── vectorstore/              # Backend-side persisted Chroma DB
│   ├── tests/
│   │   └── test_rag.py           # 7-step end-to-end pipeline test suite
│   └── app/
│       ├── __init__.py
│       ├── config.py             # Centralised config (reads from .env)
│       ├── main.py               # FastAPI app + all 5 route handlers
│       └── pipeline.py           # RAG core — embed, ingest, hybrid search, rerank, stream
│
└── frontend/                     # React 18 + Vite + TailwindCSS frontend
    ├── index.html                # Vite entry point (React app root)
    ├── package.json              # Node dependencies: react, vite, tailwindcss, lucide-react
    ├── vite.config.js            # Vite config — proxies /api/* → localhost:8000
    ├── tailwind.config.js        # TailwindCSS config with custom design tokens
    ├── postcss.config.js
    ├── public/
    │   └── favicon.svg
    └── src/
        ├── main.jsx              # ReactDOM root mount
        ├── index.css             # Global styles — glassmorphism, animations, design tokens
        ├── App.jsx               # Root layout — wires hooks to Sidebar + ChatFeed + ChatInput
        ├── components/
        │   ├── Sidebar.jsx       # Left panel: branding, status, upload, doc list, controls
        │   ├── ChatFeed.jsx      # Scrollable message history + live streaming bubble
        │   ├── ChatInput.jsx     # Auto-resizing textarea + Send / Stop buttons
        │   ├── UploadCard.jsx    # Drag-and-drop PDF uploader with status states
        │   ├── StatusBadge.jsx   # Ollama connection indicator (online / offline / loading)
        │   └── SourceChips.jsx   # Attributed document source chips below assistant messages
        └── hooks/
            ├── useChat.js        # Chat state, streaming fetch, history, abort controller
            ├── useDocuments.js   # Upload, fetch, and clear document state
            └── useStatus.js      # Polls /api/status every 15 s
```

---

## Backend — Key Files

### `backend/app/config.py`
Centralised configuration loaded from environment variables / `.env`. Defines paths, model names, RAG hyperparameters, and CORS allowed origins. All values have sensible defaults — the app works out of the box without a `.env` file.

### `backend/app/pipeline.py`
The RAG engine. All models are singletons loaded lazily at first use and cached for the lifetime of the server process. Thread-safe via a module-level lock.

| Function | Description |
|---|---|
| `bootstrap()` | Called at startup — loads persisted Chroma DB and rebuilds BM25 index from disk |
| `ingest_documents()` | Loads PDFs → splits into chunks → embeds → stores in Chroma + rebuilds BM25 |
| `hybrid_search()` | Runs vector retrieval (Chroma) + keyword retrieval (BM25), deduplicates results |
| `rerank()` | Scores candidate docs with CrossEncoder, returns top-K by relevance |
| `stream_answer()` | Async generator — builds prompt with history + context, streams tokens from Ollama, appends `\|\|\|SOURCES\|\|\|` marker |
| `get_status()` | Pings Ollama `/api/tags`, returns full health + config dict |

### `backend/app/main.py`
FastAPI application with CORS middleware and lifespan-based startup bootstrap. Handles temp file creation/cleanup for uploads and wraps `pipeline.stream_answer()` in a `StreamingResponse`.

---

## Frontend — Key Files

### `frontend/src/App.jsx`
Root React component. Composes `useStatus`, `useDocuments`, and `useChat` hooks and distributes state as props to `Sidebar`, `ChatFeed`, and `ChatInput`. Handles cross-cutting actions like clear-all (clears vectorstore + chat history + refreshes status).

### `frontend/src/hooks/useChat.js`
Manages conversation messages. On `sendMessage()`, opens a streaming fetch to `/api/chat`, accumulates tokens into `streamText`, then parses the final `|||SOURCES|||` trailer before committing the finished message to history. Supports `stopStreaming()` via an `AbortController`.

### `frontend/src/hooks/useDocuments.js`
Handles `uploadFiles()` (POST multipart to `/api/upload`), `fetchDocuments()` (GET `/api/documents`), and `clearDocuments()` (DELETE `/api/documents`). Tracks `uploadProgress` states: `uploading → processing → done | error`.

### `frontend/src/hooks/useStatus.js`
Polls `GET /api/status` every **15 seconds** and exposes Ollama health, active model name, chunk count, and pipeline config to the UI.

### `frontend/src/index.css`
Global stylesheet with full glassmorphism design system: animated gradient background, glass-card component, streaming cursor animation, source chip, drop zone, skeleton shimmer, badge variants, and custom scrollbar styling.

---

## Vite Proxy

The `vite.config.js` proxies all `/api/*` requests from the frontend dev server (port 5173) directly to the FastAPI backend (port 8000), eliminating CORS issues entirely during development:

```js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    secure: false,
  }
}
```

---

## CLI Tools

### Batch Ingestion (`backend/ingest.py`)

Ingest all PDFs from the `documents/` folder directly from the command line — no server required:

```powershell
cd backend
python ingest.py
```

Idempotent: if the vectorstore already exists, new chunks are added without rebuilding. Skips unreadable or empty files gracefully.

### RAGAS Evaluation (`backend/evaluate.py`)

Runs automated quality evaluation against a curated set of Q&A pairs using three RAGAS metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the answers grounded in the retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question asked? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |

```powershell
cd backend
pip install ragas>=0.1.7 datasets
python evaluate.py
```

Pass threshold is **0.7** per metric. Results are printed per-question and as aggregate averages.

---

## Running Tests

The test suite covers every pipeline component end-to-end:

```powershell
cd backend
python tests/test_rag.py
```

**7 test stages:**

| Stage | What's tested |
|---|---|
| 1 — Imports | All required packages importable |
| 2 — Ollama | Server reachable + phi3 inference works |
| 3 — Embeddings | `all-MiniLM-L6-v2` produces correct-dimension vectors |
| 4 — Vectorstore | Chroma DB exists, loads, and returns similarity results |
| 5 — BM25 | Index builds from stored chunks and scores queries correctly |
| 6 — Reranker | CrossEncoder ranks relevant doc above irrelevant doc |
| 7 — Full RAG | Complete retrieve → rerank → LLM pipeline returns a valid answer |

> **Note:** Run `python ingest.py` at least once before running tests so the vectorstore exists.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama offline` shown in header | Run `ollama serve` and confirm `phi3` is pulled with `ollama pull phi3` |
| Backend won't start | Check Python ≥ 3.10 and install deps: `pip install -r backend/requirements.txt` |
| Upload fails silently | Ensure the file is a valid PDF; check the backend terminal for detailed error logs |
| Same document keeps being re-uploaded | The backend deduplicates by filename. Rename the file if you need to re-ingest it |
| Chat input is disabled | At least one PDF must be uploaded and indexed before chatting |
| Frontend can't reach backend | Confirm FastAPI is running on port 8000 and you started the frontend with `npm run dev` |
| BM25 returns no results after restart | BM25 is rebuilt in-memory from Chroma on every startup — this is normal |
| RAGAS evaluation fails to import | Run `pip install ragas>=0.1.7 datasets` inside the backend virtual environment |

---

## Known Limitations

- **PDF only** — No support for DOCX, TXT, or web URLs yet
- **CPU inference** — Embedding and reranking run on CPU by default; no GPU config exposed
- **No authentication** — All API endpoints are open; add an auth layer before any public deployment
- **BM25 not persisted** — Rebuilt in-memory from Chroma on each server restart
- **Single shared vectorstore** — No per-user session isolation

---

## Roadmap

- [ ] DOCX and TXT file support
- [ ] GPU-aware embedding (CUDA device selection via env var)
- [ ] Per-document delete (remove individual files without clearing all)
- [ ] Persistent BM25 index (serialised to disk)
- [ ] Docker Compose for one-command local deployment
- [ ] User authentication (JWT / API key)
- [ ] Switchable Ollama model from the UI
- [ ] Chat export (Markdown / PDF download)
- [ ] Evaluation dashboard UI (RAGAS scores visualised)
