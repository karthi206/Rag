"""
test_rag_groq.py — End-to-end RAG pipeline test (backend/tests/test_rag_groq.py)

This mirrors test_rag.py but tests what the app ACTUALLY uses in pipeline.py:
Groq (via GROQ_API_KEY in .env) instead of a local Ollama server, and it loads
your EXISTING vectorstore instead of requiring a fresh ingest.py run.

Run from the backend/ directory: python tests/test_rag_groq.py
"""

import os
import sys
import re
import time
import traceback

os.environ["PYTHONIOENCODING"] = "utf-8"  # prevent Windows emoji encoding errors

# Make sure `app` (the backend package one level up from this tests/ folder)
# is importable no matter where this script is run from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_backend_dir, ".env"))

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results = []

def check(name, fn):
    try:
        info = fn()
        results.append((PASS, name, info or ""))
        print(f"  {PASS} {name}" + (f" — {info}" if info else ""))
        return True
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL} {name}")
        print(f"         Error: {e}")
        traceback.print_exc()
        return False

print("=" * 60)
print("  RAG PIPELINE — END-TO-END TEST (Groq)")
print("=" * 60)

# ─────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────
print(f"\n{INFO} [1/6] Import checks...")

check("rank_bm25",             lambda: __import__("rank_bm25"))
check("sentence_transformers", lambda: __import__("sentence_transformers"))
check("langchain_chroma",      lambda: __import__("langchain_chroma"))
check("langchain_huggingface", lambda: __import__("langchain_huggingface"))
check("langchain_groq",        lambda: __import__("langchain_groq"))
check("chromadb",              lambda: __import__("chromadb"))

# ─────────────────────────────────────────
# 2. GROQ API CONNECTIVITY
# ─────────────────────────────────────────
print(f"\n{INFO} [2/6] Groq API connectivity...")

from app.config import GROQ_API_KEY, MODEL_NAME, VECTORSTORE_DIR, EMBED_MODEL, RERANK_MODEL

def check_groq_key_present():
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY is empty — set it in backend/.env")
    return f"Key present (starts with '{GROQ_API_KEY[:6]}...')"

def check_groq_inference():
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY)
    t0 = time.time()
    resp = llm.invoke("What is 2+2? Reply with just the number.")
    elapsed = time.time() - t0
    text = resp.content if hasattr(resp, "content") else str(resp)
    if not text.strip():
        raise Exception("Empty response from Groq")
    return f"Model={MODEL_NAME} — Response: '{text.strip()[:60]}' in {elapsed:.1f}s"

check("GROQ_API_KEY present", check_groq_key_present)
check(f"Groq inference ({MODEL_NAME})", check_groq_inference)

# ─────────────────────────────────────────
# 3. EMBEDDING MODEL
# ─────────────────────────────────────────
print(f"\n{INFO} [3/6] Embedding model...")

from langchain_huggingface import HuggingFaceEmbeddings

def check_embeddings():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vecs = emb.embed_documents(["machine learning", "deep learning"])
    if len(vecs) != 2 or len(vecs[0]) == 0:
        raise Exception(f"Unexpected embedding output: {len(vecs)} vectors")
    return f"Dim={len(vecs[0])}, 2 vectors OK"

check("HuggingFace embed_documents", check_embeddings)

# ─────────────────────────────────────────
# 4. VECTORSTORE (existing, no fresh ingest needed)
# ─────────────────────────────────────────
print(f"\n{INFO} [4/6] Existing vectorstore...")

from langchain_chroma import Chroma

def check_vectorstore_exists():
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        raise Exception(
            f"Vectorstore not found at '{VECTORSTORE_DIR}'. "
            "If you already ingested docs elsewhere, check VECTORSTORE_DIR in .env "
            "points at the right folder."
        )
    return f"Found at '{VECTORSTORE_DIR}'"

def check_vectorstore_load():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
    count = db._collection.count()
    if count == 0:
        raise Exception("Vectorstore is empty")
    return f"{count} chunks in store"

def check_vectorstore_query():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
    docs = db.similarity_search("what is machine learning", k=3)
    if not docs:
        raise Exception("similarity_search returned 0 results")
    top = docs[0].page_content[:80].replace("\n", " ")
    return f"{len(docs)} results, top: '{top}...'"

check("Vectorstore directory exists", check_vectorstore_exists)
check("Vectorstore loads OK",         check_vectorstore_load)
check("Vectorstore similarity_search", check_vectorstore_query)

# ─────────────────────────────────────────
# 5. BM25 + CROSS-ENCODER
# ─────────────────────────────────────────
print(f"\n{INFO} [5/6] BM25 + cross-encoder reranker...")

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

def check_bm25():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
    stored = db.get()
    texts = stored.get("documents", [])
    if not texts:
        raise Exception("No documents in vectorstore to build BM25 index")
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(["machine", "learning"])
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    top_scores = [round(scores[i], 3) for i in top_idx]
    return f"Index size={len(texts)}, top BM25 scores={top_scores}"

def check_reranker():
    reranker = CrossEncoder(RERANK_MODEL)
    pairs = [
        ("what is machine learning", "Machine learning is a subset of AI."),
        ("what is machine learning", "The sky is blue on a sunny day."),
    ]
    scores = reranker.predict(pairs)
    if scores[0] <= scores[1]:
        raise Exception(f"Reranker failed: relevant ({scores[0]:.3f}) <= irrelevant ({scores[1]:.3f})")
    return f"Relevant score={scores[0]:.3f} > Irrelevant score={scores[1]:.3f} — reranking correct"

check("BM25 index build + query",           check_bm25)
check("CrossEncoder predict + ranking order", check_reranker)

# ─────────────────────────────────────────
# 6. FULL END-TO-END RAG QUERY (via pipeline.py directly)
# ─────────────────────────────────────────
print(f"\n{INFO} [6/6] Full end-to-end RAG query (via pipeline.py)...")

def check_full_rag():
    # Import the actual pipeline module so this test exercises the exact
    # same code path the FastAPI app uses — no duplicated logic to drift out of sync.
    import app.pipeline as pipeline
    pipeline.bootstrap()

    if pipeline._db is None:
        raise Exception("pipeline.bootstrap() did not find a vectorstore — check VECTORSTORE_DIR")

    import asyncio

    async def run_query():
        query = "What is machine learning?"
        chunks = []
        async for chunk in pipeline.stream_answer(query, history=[]):
            chunks.append(chunk)
        return "".join(chunks)

    t0 = time.time()
    answer = asyncio.run(run_query())
    elapsed = time.time() - t0

    # Strip the sources marker for the preview
    marker_idx = answer.find("|||SOURCES|||")
    preview_text = answer[:marker_idx] if marker_idx != -1 else answer

    if not preview_text or len(preview_text.strip()) < 5:
        raise Exception(f"Answer too short or empty: '{answer}'")

    preview = preview_text.strip()[:200].replace("\n", " ")
    return f"Answer ({elapsed:.1f}s): '{preview}...'"

check("Full RAG pipeline (pipeline.stream_answer)", check_full_rag)

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST SUMMARY")
print("=" * 60)

passed = [r for r in results if r[0] == PASS]
failed = [r for r in results if r[0] == FAIL]

for status, name, detail in results:
    print(f"  {status} {name}")

print(f"\n  Total : {len(results)}")
print(f"  Passed: {len(passed)}")
print(f"  Failed: {len(failed)}")

if failed:
    print("\n  FAILED TESTS:")
    for _, name, err in failed:
        print(f"    - {name}: {err}")
    print("\n  STATUS: NEEDS FIXES")
    sys.exit(1)
else:
    print("\n  STATUS: ALL TESTS PASSED - RAG PIPELINE READY FOR PRODUCTION")
    sys.exit(0)