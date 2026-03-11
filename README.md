# 📚 Advanced RAG Document Assistant

A **production-grade Retrieval-Augmented Generation (RAG)** application that allows you to chat with your PDF documents using a locally-running LLM. Built with **LangChain**, **ChromaDB**, **Ollama**, and **Streamlit**.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      Streamlit UI                        │
│            (Upload PDFs · Chat Interface)                │
└─────────────────────┬────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │    Document Ingestion  │
          │  PyPDFLoader → Chunk  │
          │  (RecursiveTextSplit) │
          └───────┬───────────────┘
                  │
       ┌──────────▼──────────┐
       │   Dual Index Build   │
       │  ┌────────────────┐  │
       │  │  ChromaDB      │  │  ← Dense (semantic) vector store
       │  │  (HNSWlib)     │  │
       │  └────────────────┘  │
       │  ┌────────────────┐  │
       │  │   BM25 Index   │  │  ← Sparse (keyword) index
       │  └────────────────┘  │
       └──────────┬───────────┘
                  │ Query Time
       ┌──────────▼──────────┐
       │   Hybrid Retrieval   │
       │  (Vector + BM25)    │
       └──────────┬──────────┘
                  │
       ┌──────────▼──────────┐
       │  Cross-Encoder       │
       │  Re-Ranking          │
       │  (ms-marco-MiniLM)  │
       └──────────┬──────────┘
                  │
       ┌──────────▼──────────┐
       │   Ollama LLM         │
       │  (phi3 / local)     │
       │  Streaming output   │
       └─────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔀 **Hybrid Search** | Combines BM25 (keyword) + ChromaDB (semantic) retrieval for best-of-both coverage |
| 🎯 **Cross-Encoder Re-Ranking** | `ms-marco-MiniLM-L-6-v2` re-scores retrieved chunks for higher relevance precision |
| 🌊 **Streaming Responses** | Token-by-token LLM output via Ollama for a live, responsive feel |
| 💾 **Persistent Vectorstore** | ChromaDB persists to disk — chat works on restart without re-uploading |
| 🛡️ **Deduplication Guard** | Prevents re-embedding the same file on Streamlit reruns |
| 📖 **Rolling Chat History** | Maintains last N conversation turns as context for follow-up questions |
| ⚡ **Cached Model Loading** | Embedding model, reranker, and LLM loaded once per session via `@st.cache_resource` |
| ❌ **Graceful Error Handling** | Handles corrupt PDFs, Ollama offline, and empty documents cleanly |
| 📊 **RAGAS Evaluation** | Built-in evaluation pipeline with faithfulness, answer relevancy, context precision |

---

## 🗂️ Project Structure

```
rag/
├── app.py              # Main Streamlit application
├── ingest.py           # CLI document ingestion script
├── evaluate.py         # RAGAS evaluation pipeline
├── test_rag.py         # Unit & integration tests
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your local config (not committed)
├── documents/          # Place PDFs here for CLI ingestion (not committed)
└── vectorstore/        # Auto-generated ChromaDB store (not committed)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **LLM Serving** | [Ollama](https://ollama.com/) (local, private) |
| **LLM Model** | `phi3` (default, configurable) |
| **Embedding Model** | `all-MiniLM-L6-v2` (SentenceTransformers) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) ≥ 0.4.0 |
| **Keyword Search** | [BM25Okapi](https://github.com/dorianbrown/rank_bm25) |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM Orchestration** | [LangChain](https://www.langchain.com/) |
| **PDF Loading** | PyPDFLoader |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag.git
cd rag
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### 5. Pull the LLM model via Ollama

```bash
ollama pull phi3
```

### 6. Run the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser, upload PDFs, and start chatting!

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` and adjust as needed:

```env
# LLM model name (must be pulled via: ollama pull <model>)
MODEL_NAME=phi3

# SentenceTransformer embedding model
EMBED_MODEL=all-MiniLM-L6-v2

# Directory to persist the ChromaDB vectorstore
VECTORSTORE_DIR=vectorstore

# Number of conversation turns to include as history context
MAX_HISTORY_TURNS=5

# Number of chunks to retrieve per query
K_RETRIEVE=4

# Text chunking parameters
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

---

## 📥 CLI Ingestion (`ingest.py`)

You can pre-ingest documents from the command line without the UI:

```bash
# Place your PDFs in the documents/ folder, then:
python ingest.py
```

This builds the ChromaDB vectorstore on disk so you can start the app and chat immediately without uploading.

---

## 📊 Evaluation (`evaluate.py`)

Run the RAGAS evaluation pipeline to measure RAG quality:

```bash
python evaluate.py
```

**Metrics computed:**
- **Faithfulness** — Does the answer factually match the retrieved context?
- **Answer Relevancy** — Is the answer relevant to the question asked?
- **Context Precision** — Are the retrieved chunks relevant to the question?
- **Context Recall** — Were all necessary context chunks retrieved?

---

## 🧪 Testing (`test_rag.py`)

```bash
python -m pytest test_rag.py -v
```

Covers unit tests for chunking, embedding, retrieval, hybrid search, and end-to-end pipeline integration.

---

## 🔄 How It Works — Step by Step

### Ingestion Phase
1. **PDF Loading** — `PyPDFLoader` extracts text page-by-page
2. **Chunking** — `RecursiveCharacterTextSplitter` splits into overlapping chunks (default: 300 tokens, 50 overlap)
3. **Embedding** — `all-MiniLM-L6-v2` generates 384-dim dense vectors for each chunk
4. **Indexing** — Vectors stored in ChromaDB (HNSWlib); raw text indexed in BM25

### Query Phase
1. **Dual Retrieval** — Top-K chunks via ChromaDB vector similarity + Top-K via BM25 keyword scores
2. **Fusion** — Results merged and deduplicated
3. **Re-Ranking** — Cross-encoder scores all candidates against the query; top results selected
4. **Generation** — Re-ranked context + rolling chat history injected into prompt → Ollama LLM streams the answer

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `no such table: tenants` | Delete the `vectorstore/` folder — your ChromaDB schema is outdated. Re-upload PDFs to regenerate. |
| `Ollama not running` | Start Ollama: `ollama serve` then in another terminal: `ollama pull phi3` |
| `ModuleNotFoundError: rank_bm25` | Run `pip install rank-bm25` |
| App very slow on first load | Embedding model downloads on first run (~90MB). Subsequent loads use cache. |
| Empty responses from LLM | Ensure the model is pulled: `ollama list` — if not listed, run `ollama pull phi3` |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
