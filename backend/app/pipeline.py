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
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDoc
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.config import (
    VECTORSTORE_DIR, EMBED_MODEL, RERANK_MODEL,
    MODEL_NAME, K_RETRIEVE, CHUNK_SIZE, CHUNK_OVERLAP, MAX_HISTORY_TURNS,
    PROMPT_VERSION, GROQ_API_KEY
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# Below this score, we don't trust the search results enough to let the
# AI answer. This is what stops it from guessing when it doesn't really know.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))


def load_prompt_template() -> str:
    """Reads the active prompt file from disk (see app/prompts/)."""
    path = os.path.join(PROMPTS_DIR, f"{PROMPT_VERSION}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────────────────────────────
# Singleton state (module-level, protected by a lock for thread safety)
# ─────────────────────────────────────────────────────────────────
_lock = threading.Lock()

_embeddings: Optional[HuggingFaceEmbeddings] = None
_reranker:   Optional[CrossEncoder]           = None
_llm:        Optional[ChatGroq]              = None
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


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        logger.info("Connecting to Groq model: %s", MODEL_NAME)
        _llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, streaming=True)
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
@traceable(name="hybrid_search", run_type="retriever")
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
@traceable(name="rerank", run_type="chain")
def rerank(query: str, docs: List[LCDoc], top_k: int = K_RETRIEVE) -> List[Tuple[LCDoc, float]]:
    """Returns (doc, score) pairs, best first, so callers can check confidence."""
    if not docs:
        return []
    reranker = get_reranker()
    pairs    = [(query, doc.page_content) for doc in docs]
    scores   = reranker.predict(pairs)
    ranked   = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────
# LLM streaming answer generator
# ─────────────────────────────────────────────────────────────────
@traceable(name="rag_query", run_type="chain")
async def stream_answer(query: str, history: List[Tuple[str, str]]) -> AsyncGenerator[str, None]:
    raw_docs   = hybrid_search(query)
    ranked     = rerank(query, raw_docs)  # list of (doc, score)
    raw_docs   = hybrid_search(query, k=15)      # wide candidate pool
    ranked     = rerank(query, raw_docs, top_k=K_RETRIEVE)  # narrow to 4 for the LLM
    # ── Real citation enforcement ──────────────────────────────
    # If nothing came back, or even our best match scored too low to trust,
    # refuse to answer instead of letting the AI guess. This does not rely
    # on the AI "choosing" to follow instructions.
    best_score = ranked[0][1] if ranked else None
    answered   = bool(ranked) and not (best_score is not None and best_score < MIN_RERANK_SCORE)

    # Tag this trace with RAG-specific metadata Langsmith wouldn't know
    # about automatically — this is what lets us compute "citation
    # coverage" (% of queries actually answered vs refused) later.
    run = get_current_run_tree()
    if run is not None:
        run.extra = run.extra or {}
        run.extra["metadata"] = {
            **(run.extra.get("metadata") or {}),
            "answered":             answered,
            "best_rerank_score":    float(best_score) if best_score is not None else None,
            "num_chunks_retrieved": len(raw_docs),
        }

    if not ranked or best_score < MIN_RERANK_SCORE:
        logger.info(
            "Refusing to answer — best match score %s below threshold %s",
            best_score, MIN_RERANK_SCORE
        )
        yield "I cannot find the answer in the provided documents."
        return

    docs = [doc for doc, _ in ranked]
    context = "\n\n".join(doc.page_content for doc in docs)

    sources = set()
    for doc in docs:
        page   = doc.metadata.get("page")
        source = doc.metadata.get("source", "document")
        if page is not None:
            sources.add(f"{source} — Page {int(page) + 1}")

    if run is not None:
        run.extra["metadata"]["num_sources_cited"] = len(sources)

    recent_history = history[-(MAX_HISTORY_TURNS * 2):]
    history_text   = "".join(f"{r}: {m}\n" for r, m in recent_history)

    prompt = load_prompt_template().format(
        max_history_turns=MAX_HISTORY_TURNS,
        history_text=history_text,
        context=context,
        query=query,
    )

    llm = get_llm()

    # Stream tokens
    for chunk in llm.stream(prompt):
        text = chunk.content if hasattr(chunk, "content") else chunk
        if text:
            yield text

    # After streaming completes, yield a special sources marker
    if sources:
        sources_str = "|||SOURCES|||" + "|||".join(sorted(sources))
        yield sources_str


# ─────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────
def get_status() -> dict:
    llm_online = bool(GROQ_API_KEY)

    return {
        "llm_online":     llm_online,
        "llm_provider":   "groq",
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


# ─────────────────────────────────────────────────────────────────
# Clear vectorstore
# ─────────────────────────────────────────────────────────────────
def clear_vectorstore() -> None:
    """
    Deletes the Chroma collection via its own API (which safely closes
    its internal HNSW index) instead of deleting files on disk directly.
    Avoids Windows file-lock issues entirely.
    """
    global _db, _bm25, _splits, _ingested_sources

    with _lock:
        if _db is not None:
            try:
                _db.delete_collection()
            except Exception as exc:
                logger.warning("delete_collection failed (continuing): %s", exc)
            finally:
                _db = None

        _bm25             = None
        _splits           = []
        _ingested_sources = set()

@traceable(name="hybrid_search", run_type="retriever")
def hybrid_search(query: str, k: int = 15) -> List[LCDoc]:
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

    return unique_docs  # don't truncate to k here — let rerank() do the final cut