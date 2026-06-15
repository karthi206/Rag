"""
pipeline.py — Modular RAG core for the FastAPI backend.

Responsibilities:
  - Load / cache embedding model, reranker, and LLM once at startup
  - Bootstrap Chroma vectorstore and BM25 index from disk
  - ingest_documents() — chunk + embed + store new PDFs
  - hybrid_search()    — Chroma vector + BM25 keyword fusion
  - rerank()           — cross-encoder re-scoring
  - stream_answer()    — async generator yielding LLM tokens
"""

import os
import re
import threading
import logging
from typing import AsyncGenerator, List, Tuple, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDoc
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import (
    VECTORSTORE_DIR, EMBED_MODEL, RERANK_MODEL,
    MODEL_NAME, K_RETRIEVE, CHUNK_SIZE, CHUNK_OVERLAP, MAX_HISTORY_TURNS
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Singleton state (module-level, protected by a lock for thread safety)
# ─────────────────────────────────────────────────────────────────
_lock = threading.Lock()

_embeddings: Optional[HuggingFaceEmbeddings] = None
_reranker:   Optional[CrossEncoder]           = None
_llm:        Optional[OllamaLLM]              = None
_db:         Optional[Chroma]                         = None
_bm25:       Optional[BM25Okapi]                      = None
_splits:     List[LCDoc]                          = []
_ingested_sources: set                            = set()


# ─────────────────────────────────────────────────────────────────
# Lazy model loaders
# ─────────────────────────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker: %s", RERANK_MODEL)
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        logger.info("Connecting to Ollama model: %s", MODEL_NAME)
        _llm = OllamaLLM(model=MODEL_NAME)
    return _llm


# ─────────────────────────────────────────────────────────────────
# Bootstrap — load existing vectorstore + BM25 from disk on startup
# ─────────────────────────────────────────────────────────────────
def bootstrap() -> None:
    """Called once at server startup to load persisted data."""
    global _db, _bm25, _splits, _ingested_sources

    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        try:
            logger.info("Bootstrapping vectorstore from disk: %s", VECTORSTORE_DIR)
            _db = Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=get_embeddings()
            )
            stored = _db.get()
            if stored and stored.get("documents"):
                docs_text = stored["documents"]
                metas     = stored.get("metadatas") or [{}] * len(docs_text)
                _splits   = [LCDoc(page_content=t, metadata=m) for t, m in zip(docs_text, metas)]
                tokenized = [re.findall(r"\w+", t.lower()) for t in docs_text]
                _bm25     = BM25Okapi(tokenized)

                # Populate set of already-ingested filenames
                for m in metas:
                    src = m.get("source")
                    if src:
                        _ingested_sources.add(src)

                logger.info("Bootstrapped %d chunks from disk.", len(_splits))
        except Exception as exc:
            logger.warning("Could not load existing vectorstore: %s", exc)
    else:
        logger.info("No existing vectorstore found. Ready for first upload.")


# ─────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────
def ingest_documents(file_paths: List[Tuple[str, str]]) -> dict:
    """
    Ingest a list of (temp_file_path, original_filename) pairs.
    Returns summary dict with counts.
    """
    global _db, _bm25, _splits, _ingested_sources

    all_docs   = []
    loaded     = []
    skipped    = []
    failed     = []

    for temp_path, original_name in file_paths:
        if original_name in _ingested_sources:
            skipped.append(original_name)
            continue
        try:
            loader = PyPDFLoader(temp_path)
            docs   = loader.load()
            if not docs:
                failed.append(original_name)
                continue
            for doc in docs:
                doc.metadata["source"] = original_name
            all_docs.extend(docs)
            loaded.append(original_name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", original_name, exc)
            failed.append(original_name)

    if not all_docs:
        return {"loaded": loaded, "skipped": skipped, "failed": failed, "chunks": 0}

    splitter   = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_splits = splitter.split_documents(all_docs)

    with _lock:
        emb = get_embeddings()
        if _db is None:
            if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
                _db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
                _db.add_documents(new_splits)
            else:
                _db = Chroma.from_documents(new_splits, emb, persist_directory=VECTORSTORE_DIR)
        else:
            _db.add_documents(new_splits)

        _splits.extend(new_splits)
        chunk_texts = [doc.page_content for doc in _splits]
        tokenized   = [re.findall(r"\w+", t.lower()) for t in chunk_texts]
        _bm25       = BM25Okapi(tokenized)

        for name in loaded:
            _ingested_sources.add(name)

    return {
        "loaded":  loaded,
        "skipped": skipped,
        "failed":  failed,
        "chunks":  len(new_splits),
        "total_chunks": len(_splits),
    }


# ─────────────────────────────────────────────────────────────────
# Hybrid search
# ─────────────────────────────────────────────────────────────────
def hybrid_search(query: str, k: int = K_RETRIEVE) -> List[LCDoc]:
    if _db is None:
        return []

    retriever      = _db.as_retriever(search_kwargs={"k": k})
    vector_results = retriever.invoke(query)

    tokenized_query = re.findall(r"\w+", query.lower())
    if not tokenized_query or _bm25 is None:
        return vector_results[:k]

    bm25_scores         = _bm25.get_scores(tokenized_query)
    top_keyword_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    keyword_results     = [_splits[i] for i in top_keyword_indices if i < len(_splits)]

    combined = vector_results + keyword_results
    seen, unique_docs = set(), []
    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:k]


# ─────────────────────────────────────────────────────────────────
# Re-ranking
# ─────────────────────────────────────────────────────────────────
def rerank(query: str, docs: List[LCDoc], top_k: int = K_RETRIEVE) -> List[LCDoc]:
    if not docs:
        return []
    reranker = get_reranker()
    pairs    = [(query, doc.page_content) for doc in docs]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────
# LLM streaming answer generator
# ─────────────────────────────────────────────────────────────────
async def stream_answer(query: str, history: List[Tuple[str, str]]) -> AsyncGenerator[str, None]:
    docs = hybrid_search(query)
    docs = rerank(query, docs)

    context = "\n\n".join(doc.page_content for doc in docs)

    sources = set()
    for doc in docs:
        page   = doc.metadata.get("page")
        source = doc.metadata.get("source", "document")
        if page is not None:
            sources.add(f"{source} — Page {int(page) + 1}")

    recent_history = history[-(MAX_HISTORY_TURNS * 2):]
    history_text   = "".join(f"{r}: {m}\n" for r, m in recent_history)

    prompt = f"""You are a precise document assistant. Answer ONLY from the context below.
If the answer is not in the documents, say: "I cannot find the answer in the provided documents."
Be concise and factual. Do not make up information.

Conversation history (last {MAX_HISTORY_TURNS} turns):
{history_text}
Document context:
{context}

Question: {query}

Answer:"""

    llm = get_llm()

    # Stream tokens
    for chunk in llm.stream(prompt):
        if chunk:
            yield chunk

    # After streaming completes, yield a special sources marker
    if sources:
        sources_str = "|||SOURCES|||" + "|||".join(sorted(sources))
        yield sources_str


# ─────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────
def get_status() -> dict:
    import urllib.request, json as _json
    ollama_ok  = False
    ollama_models = []
    try:
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = _json.loads(r.read())
        ollama_models = [m["name"] for m in data.get("models", [])]
        ollama_ok = True
    except Exception:
        pass

    return {
        "ollama":         ollama_ok,
        "ollama_models":  ollama_models,
        "active_model":   MODEL_NAME,
        "embed_model":    EMBED_MODEL,
        "rerank_model":   RERANK_MODEL,
        "chunks_indexed": len(_splits),
        "documents":      sorted(_ingested_sources),
        "vectorstore":    VECTORSTORE_DIR,
        "k_retrieve":     K_RETRIEVE,
        "chunk_size":     CHUNK_SIZE,
        "chunk_overlap":  CHUNK_OVERLAP,
    }
