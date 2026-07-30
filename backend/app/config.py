"""
config.py — Centralized configuration for the FastAPI RAG backend.
All values can be overridden via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vs_dir = os.getenv("VECTORSTORE_DIR", "vectorstore")
VECTORSTORE_DIR = _vs_dir if os.path.isabs(_vs_dir) else os.path.join(BASE_DIR, _vs_dir)

_docs_dir = os.getenv("DOCUMENTS_DIR", "documents")
DOCUMENTS_DIR = _docs_dir if os.path.isabs(_docs_dir) else os.path.join(BASE_DIR, _docs_dir)

# ─── Model settings ───────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME",   "phi3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_BASE  = os.getenv("OLLAMA_BASE",  "http://localhost:11434")

# ─── Prompts ────────────────────────────────────────────────────
# Which prompt file (in app/prompts/) to use. Change this to try a new
# prompt without touching any pipeline code — just add a new file like
# qa_prompt_v2.txt and point this at it.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "qa_prompt_v1")

# ─── RAG hyper-parameters ─────────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
K_RETRIEVE        = int(os.getenv("K_RETRIEVE",        "4"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",        "700"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP",     "100"))

# ─── CORS ─────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Ensure storage directories exist
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR,   exist_ok=True)
