"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /api/status       — Health check (Ollama, vectorstore stats)
  GET  /api/documents    — List all ingested documents
  POST /api/upload       — Upload one or more PDFs for ingestion
  POST /api/chat         — Send a query and stream the response (SSE)
  DELETE /api/documents  — Clear all vectorstore data
"""

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import app.pipeline as pipeline
from app.config import CORS_ORIGINS, VECTORSTORE_DIR

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Lifespan — replaces deprecated @app.on_event("startup")
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    logger.info("Server startup: bootstrapping RAG pipeline…")
    pipeline.bootstrap()
    logger.info("Bootstrap complete.")
    yield


# ─────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Assistant API",
    description="Production-grade RAG backend: Hybrid BM25 + Chroma, Cross-Encoder reranking, Ollama LLM streaming.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# Schema models
# ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:   str
    history: List[List[str]] = []  # [["user", "..."], ["assistant", "..."]]


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Returns system health: Ollama connectivity, model names, chunk counts."""
    return pipeline.get_status()


@app.get("/api/documents")
async def list_documents():
    """Returns list of all ingested document filenames."""
    status = pipeline.get_status()
    return {"documents": status["documents"], "total_chunks": status["chunks_indexed"]}


@app.delete("/api/documents")
async def clear_documents():
    """Clears the vectorstore and resets the pipeline state."""
    try:
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)

        # Reset in-memory state
        pipeline._db               = None
        pipeline._bm25             = None
        pipeline._splits           = []
        pipeline._ingested_sources = set()

        return {"message": "Vectorstore cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Accept one or more PDF uploads, ingest them into the RAG pipeline,
    and return a summary of the operation.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    temp_pairs = []
    cleanup    = []

    for uploaded in files:
        if not uploaded.filename.lower().endswith(".pdf"):
            continue
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            content = await uploaded.read()
            tf.write(content)
            tf.close()
            temp_pairs.append((tf.name, uploaded.filename))
            cleanup.append(tf.name)
        except Exception as exc:
            logger.error("Error saving upload %s: %s", uploaded.filename, exc)

    if not temp_pairs:
        raise HTTPException(status_code=400, detail="No valid PDF files found in the upload.")

    try:
        result = pipeline.ingest_documents(temp_pairs)
    finally:
        for path in cleanup:
            try:
                os.unlink(path)
            except Exception:
                pass

    return result


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Accepts a query and conversation history, runs the full RAG pipeline,
    and streams the LLM answer back as plain text chunks.
    Sources are appended as a special marker at the end of the stream.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if pipeline._db is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please upload PDFs first."
        )

    history_pairs = [tuple(pair) for pair in request.history if len(pair) == 2]

    async def token_stream():
        try:
            async for chunk in pipeline.stream_answer(request.query, history_pairs):
                yield chunk
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"\n\n⚠️ Error: {exc}"

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
        headers={"X-Content-Type-Options": "nosniff"},
    )


# ─────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "RAG Document Assistant API v2.0 — see /docs for Swagger UI"}
