# RAG Document Assistant

> A full-stack, locally-running **Retrieval-Augmented Generation (RAG)** system — ask questions about your own PDF documents using hybrid search, cross-encoder reranking, and a local LLM via Ollama.

---

## Architecture Overview

```
┌─────────────────────────────┐        REST / Streaming HTTP
│  React Frontend  (port 5173) │ ◄──────────────────────────────► ┌──────────────────────────────┐
│  Vite · TailwindCSS          │                                   │  FastAPI Backend  (port 8000) │
│  index.html (standalone)     │                                   │  uvicorn · python-multipart   │
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

- **Hybrid Search** — Combines dense vector similarity (Chroma + `all-MiniLM-L6-v2`) with sparse keyword matching (BM25) for maximum recall.
- **Cross-Encoder Reranking** — Retrieved candidates are rescored by `ms-marco-MiniLM-L-6-v2` for precision.
- **Streaming LLM Responses** — Token-by-token streaming via Ollama (`phi3`) with a live blinking cursor.
- **Source Attribution** — Every answer includes cited document & page references parsed from the stream.
- **Conversational Memory** — Rolling multi-turn chat history (last 5 turns) sent with every query for context.
- **Persistent Vectorstore** — Chroma DB is persisted to disk; previous documents survive server restarts.
- **Deduplication Guard** — Re-uploading the same filename is silently skipped, preventing duplicate chunks.
- **Drag-and-Drop Upload** — Browser-native PDF drag-and-drop with real-time progress indicator.
- **System Status Polling** — Frontend polls `/api/status` every 15 seconds for live Ollama health, chunk counts, and model info.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 + | Python 3.14 works with warnings |
| Node.js | 20 + | For the React frontend |
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

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health — Ollama state, active model, chunk count, config |
| `GET` | `/api/documents` | List all ingested document filenames |
| `POST` | `/api/upload` | Upload one or more PDF files for ingestion |
| `POST` | `/api/chat` | Send a query + history; streams plain-text tokens |
| `DELETE` | `/api/documents` | Wipe the entire vectorstore and reset pipeline state |

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
  "loaded":  ["file1.pdf"],
  "skipped": [],
  "failed":  [],
  "chunks":  42,
  "total_chunks": 182
}
```

### Chat Response

Plain-text streaming response. Sources are appended as a trailer at the end of the stream:

```
...last token of the answer|||SOURCES|||filename.pdf — Page 3|||filename.pdf — Page 7
```

---

## Configuration

Create a `.env` file in the `backend/` directory (a template `.env` is already included):

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
├── app.py                        # Legacy Streamlit single-file app (standalone)
├── ingest.py                     # Standalone CLI ingestion script
├── evaluate.py                   # RAG evaluation harness
├── test_rag.py                   # Unit & integration tests
├── requirements.txt              # Root-level Python deps (Streamlit app)
├── README.md
│
├── documents/                    # Drop PDFs here for bulk ingestion
│   ├── data science.pdf
│   ├── ml.pdf
│   └── sample.pdf
│
├── vectorstore/                  # Persisted Chroma DB (auto-created)
│
├── backend/                      # FastAPI production backend
│   ├── run.py                    # Dev server launcher (uvicorn)
│   ├── requirements.txt          # Backend Python dependencies
│   ├── .env                      # Environment configuration
│   ├── documents/                # Backend-side document storage
│   └── app/
│       ├── __init__.py
│       ├── config.py             # Centralised config (reads from .env)
│       ├── main.py               # FastAPI app + all route handlers
│       └── pipeline.py           # RAG core — embed, ingest, search, rerank, stream
│
└── frontend/                     # React + Vite + TailwindCSS frontend
    ├── index.html                # Standalone vanilla HTML/CSS/JS UI (entry point)
    ├── package.json
    ├── vite.config.js            # Vite config — proxies /api/* → :8000
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── public/
    │   └── favicon.svg
    └── src/                      # React component tree (alternative UI)
        ├── main.jsx              # ReactDOM root mount
        ├── App.jsx               # Root layout — wires hooks to components
        ├── index.css             # Global styles + TailwindCSS + design tokens
        ├── components/
        │   ├── Sidebar.jsx       # Left panel: branding, status, upload, doc list, controls
        │   ├── ChatFeed.jsx      # Scrollable message history + live streaming bubble
        │   ├── ChatInput.jsx     # Auto-resizing textarea + send / stop buttons
        │   ├── UploadCard.jsx    # Drag-and-drop PDF uploader with status states
        │   ├── StatusBadge.jsx   # Ollama connection indicator badge
        │   └── SourceChips.jsx   # Attributed document source chips
        └── hooks/
            ├── useChat.js        # Chat state, streaming, history, abort controller
            ├── useDocuments.js   # Upload, fetch, and clear document state
            └── useStatus.js      # Polls /api/status every 15 s
```

---

## Backend — Key Files

### `backend/app/config.py`
Centralised configuration loaded from environment variables / `.env`. Defines paths, model names, RAG hyperparameters, and CORS allowed origins.

### `backend/app/pipeline.py`
The RAG engine. All models are singletons loaded lazily at first use and cached for the lifetime of the server process.

| Function | Description |
|---|---|
| `bootstrap()` | Loads persisted Chroma DB + rebuilds BM25 index from disk at startup |
| `ingest_documents()` | Loads PDFs → splits into chunks → embeds → stores in Chroma + BM25 |
| `hybrid_search()` | Runs vector retrieval (Chroma) + keyword retrieval (BM25), deduplicates |
| `rerank()` | Scores candidate docs with cross-encoder, returns top-K |
| `stream_answer()` | Async generator: builds prompt with history + context, streams tokens from Ollama, appends `|||SOURCES|||` marker |
| `get_status()` | Pings Ollama `/api/tags`, returns health dict |

### `backend/app/main.py`
FastAPI application with CORS middleware. Handles file I/O (temp file creation/cleanup) for uploads and wraps `pipeline.stream_answer()` in a `StreamingResponse`.

---

## Frontend — Key Files

### `frontend/index.html`
Self-contained vanilla HTML/CSS/JS single-page application. Handles all API communication, drag-and-drop file selection, live streaming response decoding, source chip rendering, and a lightweight Markdown formatter — all without any build step required.

### `frontend/src/App.jsx`
Root React component. Composes `useStatus`, `useDocuments`, and `useChat` hooks and distributes state as props down to `Sidebar`, `ChatFeed`, and `ChatInput`.

### `frontend/src/hooks/useChat.js`
Manages conversation messages. On `sendMessage()`, opens a streaming fetch to `/api/chat`, accumulates tokens into `streamText`, then parses the final `|||SOURCES|||` trailer before committing the finished message to history.

### `frontend/src/hooks/useDocuments.js`
Handles `uploadFiles()` (POST multipart to `/api/upload`), `fetchDocuments()` (GET `/api/documents`), and `clearDocuments()` (DELETE `/api/documents`).

### `frontend/src/hooks/useStatus.js`
Polls `GET /api/status` every **15 seconds** and exposes Ollama health, active model name, chunk count, and pipeline config to the UI.

---

## Vite Proxy

The `vite.config.js` proxies all `/api/*` requests from the frontend dev server (port 5173) directly to the FastAPI backend (port 8000), eliminating CORS issues entirely during development:

```js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  }
}
```

---

## Legacy / Standalone Files

| File | Purpose |
|---|---|
| `app.py` | Original Streamlit single-file RAG app — runs independently without the backend/frontend split |
| `ingest.py` | CLI script to bulk-ingest PDFs from the `documents/` folder directly |
| `evaluate.py` | Evaluates RAG answer quality against a set of ground-truth Q&A pairs |
| `test_rag.py` | Unit and integration tests for the RAG pipeline |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama offline` shown in header | Run `ollama serve` and confirm `phi3` is pulled |
| Backend won't start | Check Python version ≥ 3.10; install deps with `pip install -r backend/requirements.txt` |
| Upload fails silently | Ensure the file is a valid PDF; check backend terminal for detailed error logs |
| Same document keeps being re-uploaded | The backend deduplicates by filename. Rename the file if you need to re-ingest |
| Chat input disabled | Documents must be loaded first — upload at least one PDF |
| Frontend can't reach backend | Confirm FastAPI is running on port 8000 and Vite proxy is active (`npm run dev`) |
