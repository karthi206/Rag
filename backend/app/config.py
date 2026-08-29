"""
config.py — Centralized configuration for the FastAPI RAG backend.
All values can be overridden via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
<<<<<<< HEAD
_vs_dir = os.getenv("VECTORSTORE_DIR", "vectorstore")
VECTORSTORE_DIR = _vs_dir if os.path.isabs(_vs_dir) else os.path.join(BASE_DIR, _vs_dir)

_docs_dir = os.getenv("DOCUMENTS_DIR", "documents")
DOCUMENTS_DIR = _docs_dir if os.path.isabs(_docs_dir) else os.path.join(BASE_DIR, _docs_dir)

# ─── Model settings ───────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME",   "phi3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
=======
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(BASE_DIR, "vectorstore"))
DOCUMENTS_DIR   = os.getenv("DOCUMENTS_DIR",   os.path.join(BASE_DIR, "documents"))

# ─── Model settings ───────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME",   "phi3")
>>>>>>> 4218456db65b5d66eb915e93a076f44adecf548f
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_BASE  = os.getenv("OLLAMA_BASE",  "http://localhost:11434")

<<<<<<< HEAD
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
K_RETRIEVE_CANDIDATES = int(os.getenv("K_RETRIEVE_CANDIDATES",     "15"))
K_RETRIEVE_FINAL     = int(os.getenv("K_RETRIEVE_FINAL",     "4"))
=======
# ─── RAG hyper-parameters ─────────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
K_RETRIEVE        = int(os.getenv("K_RETRIEVE",        "4"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",        "300"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP",     "50"))
>>>>>>> 4218456db65b5d66eb915e93a076f44adecf548f

# ─── CORS ─────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Ensure storage directories exist
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR,   exist_ok=True)
