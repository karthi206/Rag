# RAG Document Assistant — Project Info

> A production-grade Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDFs and chat with them using a local LLM.

---

## 1. Project Overview

**Final Project Name:** RAG Document Assistant

**One-line description:**
> Upload your PDFs and chat with them — powered by hybrid search, cross-encoder reranking, and a local Ollama LLM, all running 100% offline.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (via HuggingFace / SentenceTransformers) |
| **Vector Store** | ChromaDB (persistent, local) |
| **Keyword Search** | BM25 (`rank-bm25` / BM25Okapi) |
| **Reranker** | CrossEncoder — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Ollama — `phi3` (local, no API key needed) |
| **Backend Framework** | FastAPI + Uvicorn |
| **Frontend Framework** | React 18 + Vite + TailwindCSS |
| **PDF Parsing** | PyPDF (via LangChain `PyPDFLoader`) |
| **LangChain** | `langchain`, `langchain-chroma`, `langchain-ollama`, `langchain-huggingface` |

> ✅ No changes from expected stack — everything confirmed from codebase.

---

## 3. Features — What's Working Now

- ✅ **PDF Upload** — Upload one or more PDFs via the UI or API; auto-ingested into the pipeline
- ✅ **Hybrid Search** — Combines ChromaDB vector search + BM25 keyword search for better recall
- ✅ **Cross-Encoder Reranking** — Re-scores retrieved chunks for higher relevance precision
- ✅ **Streaming Chat** — LLM response streamed token-by-token via Server-Sent Events (SSE)
- ✅ **Conversation History** — Maintains last 5 turns of context in the prompt
- ✅ **Source Attribution** — Shows which document + page the answer came from
- ✅ **Duplicate Prevention** — Skips re-ingesting already-indexed documents
- ✅ **Document Management** — List all ingested docs, clear the vectorstore via API
- ✅ **Health Status** — `/api/status` endpoint reports Ollama connectivity, model info, chunk counts
- ✅ **Persistent Vectorstore** — ChromaDB persisted to disk; survives server restarts
- ✅ **Offline / Local** — Entire pipeline runs locally, no external API keys required

---

## 4. What's in This Repo

### Project Architecture

```
Rag-master/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI app, routes, lifespan
│   │   ├── pipeline.py    # Core RAG logic (embed, search, rerank, stream)
│   │   └── config.py      # Centralized config (env-overridable)
│   ├── ingest.py          # Standalone CLI ingestion script
│   ├── evaluate.py        # Evaluation / benchmarking script
│   ├── run.py             # Dev server launcher (uvicorn)
│   ├── requirements.txt   # Python dependencies
│   └── .env.example       # Example environment variables
├── frontend/
│   ├── src/               # React components
│   ├── index.html         # Single-page app entry
│   ├── package.json       # Node dependencies (React, Vite, TailwindCSS)
│   └── vite.config.js     # Vite config
├── documents/             # Drop PDFs here for batch ingestion
├── vectorstore/           # ChromaDB persisted data (auto-created)
└── README.md
```

### RAG Pipeline Flow

```
User Query
    │
    ▼
Hybrid Search ──────────────────────────────────────┐
    ├── ChromaDB Vector Search (semantic)            │
    └── BM25 Keyword Search (lexical)                │
                                                     │
Combined & Deduplicated Results ◄────────────────────┘
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Top-K Chunks → Prompt Construction (+ conversation history)
    │
    ▼
Ollama phi3 LLM (streaming)
    │
    ▼
Streamed Answer + Source Attribution → Frontend UI
```

---

## 5. Setup Instructions — Run Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- `phi3` model pulled in Ollama

```bash
# Pull the LLM model
ollama pull phi3
```

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env if needed (model names, ports, paths)

# Start the FastAPI server
python run.py
# Server runs at: http://localhost:8000
# Swagger docs at: http://localhost:8000/docs
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
# App runs at: http://localhost:5173
```

---

## 6. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Health check — Ollama status, model info, chunk counts |
| `GET` | `/api/documents` | List all ingested document filenames |
| `POST` | `/api/upload` | Upload one or more PDF files for ingestion |
| `POST` | `/api/chat` | Send a query + history; streams LLM answer as plain text |
| `DELETE` | `/api/documents` | Clear all vectorstore data and reset pipeline state |
| `GET` | `/docs` | Auto-generated Swagger UI (FastAPI built-in) |

### Example — Chat Request

```json
POST /api/chat
{
  "query": "What is the main conclusion of the paper?",
  "history": [
    ["user", "What is this document about?"],
    ["assistant", "It is about ..."]
  ]
}
```

### Example — Upload

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "files=@my_document.pdf"
```

---

## 7. Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `phi3` | Ollama LLM model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder reranker |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama base URL |
| `VECTORSTORE_DIR` | `./vectorstore` | Path to persist ChromaDB |
| `DOCUMENTS_DIR` | `./documents` | Path for batch PDF ingestion |
| `CHUNK_SIZE` | `300` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `K_RETRIEVE` | `4` | Number of top chunks to retrieve & rerank |
| `MAX_HISTORY_TURNS` | `5` | Max conversation turns kept in prompt |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins |

---

## 8. GitHub Repository

**Repo:** [karthi206/Rag](https://github.com/karthi206/Rag)

---

## 9. Known Limitations & Pending Things

### Current Limitations
- 📄 **PDF only** — No support for DOCX, TXT, or web URLs yet
- 🧠 **No GPU acceleration config** — Embedding/reranking runs on CPU by default (can be slow for large corpora)
- 🔒 **No authentication** — API endpoints are open; not production-ready without auth layer
- 🔄 **BM25 is rebuilt in-memory** — Not persisted to disk separately; rebuilt from ChromaDB on each server restart
- 📦 **Single-user** — No multi-user session isolation; shared vectorstore state across all requests

### Pending / Future Plans
- [ ] DOCX and TXT file support
- [ ] GPU-aware embedding (CUDA device config)
- [ ] User authentication (JWT / API keys)
- [ ] Persistent BM25 index (pickle to disk)
- [ ] Document-level delete (remove individual files, not just full clear)
- [ ] Evaluation dashboard (RAGAS metrics — `evaluate.py` already scaffolded)
- [ ] Docker Compose setup for one-command deployment
- [ ] Support for multiple Ollama models (switchable from UI)
- [ ] Chat export (download conversation as PDF/Markdown)

---

## 10. Screenshots

> 📸 _Screenshots to be added — run the app locally and capture the chat UI, upload flow, and status panel._

---

*Last updated: June 2026*
