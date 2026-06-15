"""
config.py — Centralized configuration for the FastAPI RAG backend.
All values can be overridden via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(BASE_DIR, "vectorstore"))
DOCUMENTS_DIR   = os.getenv("DOCUMENTS_DIR",   os.path.join(BASE_DIR, "documents"))

# ─── Model settings ───────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME",   "phi3")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_BASE  = os.getenv("OLLAMA_BASE",  "http://localhost:11434")

# ─── RAG hyper-parameters ─────────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))
K_RETRIEVE        = int(os.getenv("K_RETRIEVE",        "4"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",        "300"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP",     "50"))

# ─── CORS ─────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Ensure storage directories exist
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR,   exist_ok=True)
