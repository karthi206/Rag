"""
test_rag.py — End-to-end RAG pipeline test (backend/tests/test_rag.py)
Tests every component in sequence and prints a clear PASS/FAIL report.
Run from the backend/ directory: python tests/test_rag.py
"""

import os
import sys
import re
import time
import traceback

os.environ["PYTHONIOENCODING"] = "utf-8"  # prevent Windows emoji encoding errors

from dotenv import load_dotenv
load_dotenv()

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
WARN = "[WARN]"

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
print("  RAG PIPELINE — END-TO-END TEST")
print("=" * 60)

# ─────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────
print(f"\n{INFO} [1/7] Import checks...")

check("rank_bm25",           lambda: __import__("rank_bm25"))
check("sentence_transformers", lambda: __import__("sentence_transformers"))
check("langchain_community",  lambda: __import__("langchain_community"))
check("langchain_huggingface", lambda: __import__("langchain_huggingface"))
check("chromadb",             lambda: __import__("chromadb"))
check("ragas",                lambda: __import__("ragas"))

# ─────────────────────────────────────────
# 2. OLLAMA CONNECTIVITY
# ─────────────────────────────────────────
print(f"\n{INFO} [2/7] Ollama connectivity...")

import urllib.request, json

def check_ollama():
    r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    data = json.loads(r.read())
    models = [m["name"] for m in data.get("models", [])]
    if not models:
        raise Exception("No models found in Ollama")
    return f"Models available: {', '.join(models)}"

def check_phi3_inference():
    from langchain_community.llms import Ollama
    llm = Ollama(model="phi3")
    t0 = time.time()
    resp = llm.invoke("What is 2+2? Reply with just the number.")
    elapsed = time.time() - t0
    if not resp.strip():
        raise Exception("Empty response from phi3")
    return f"Response: '{resp.strip()[:60]}' in {elapsed:.1f}s"

check("Ollama server reachable", check_ollama)
check("phi3 inference (LLM call)", check_phi3_inference)

# ─────────────────────────────────────────
# 3. EMBEDDING MODEL
# ─────────────────────────────────────────
print(f"\n{INFO} [3/7] Embedding model...")

from langchain_community.embeddings import SentenceTransformerEmbeddings

def check_embeddings():
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vecs = emb.embed_documents(["machine learning", "deep learning"])
    if len(vecs) != 2 or len(vecs[0]) == 0:
        raise Exception(f"Unexpected embedding output: {len(vecs)} vectors")
    return f"Dim={len(vecs[0])}, 2 vectors OK"

check("SentenceTransformer embed_documents", check_embeddings)

# ─────────────────────────────────────────
# 4. VECTORSTORE
# ─────────────────────────────────────────
print(f"\n{INFO} [4/7] Vectorstore...")

from langchain_community.vectorstores import Chroma

VECTORSTORE_DIR = "vectorstore"

def check_vectorstore_exists():
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        raise Exception(f"Vectorstore not found at '{VECTORSTORE_DIR}'. Run ingest.py first.")
    return f"Found at '{VECTORSTORE_DIR}'"

def check_vectorstore_load():
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
    count = db._collection.count()
    if count == 0:
        raise Exception("Vectorstore is empty — run ingest.py first")
    return f"{count} chunks in store"

def check_vectorstore_query():
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
    docs = db.similarity_search("what is machine learning", k=3)
    if not docs:
        raise Exception("similarity_search returned 0 results")
    top = docs[0].page_content[:80].replace("\n", " ")
    return f"{len(docs)} results, top: '{top}...'"

check("Vectorstore directory exists", check_vectorstore_exists)
check("Vectorstore loads OK", check_vectorstore_load)
check("Vectorstore similarity_search", check_vectorstore_query)

# ─────────────────────────────────────────
# 5. BM25 RETRIEVAL
# ─────────────────────────────────────────
print(f"\n{INFO} [5/7] BM25 keyword retrieval...")

from rank_bm25 import BM25Okapi

def check_bm25():
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
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

check("BM25 index build + query", check_bm25)

# ─────────────────────────────────────────
# 6. CROSS-ENCODER RERANKER
# ─────────────────────────────────────────
print(f"\n{INFO} [6/7] Cross-encoder reranker...")

from sentence_transformers import CrossEncoder

def check_reranker():
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [
        ("what is machine learning", "Machine learning is a subset of AI."),
        ("what is machine learning", "The sky is blue on a sunny day."),
    ]
    scores = reranker.predict(pairs)
    if scores[0] <= scores[1]:
        raise Exception(f"Reranker failed: relevant doc score ({scores[0]:.3f}) <= irrelevant ({scores[1]:.3f})")
    return f"Relevant score={scores[0]:.3f} > Irrelevant score={scores[1]:.3f} — reranking correct"

check("CrossEncoder predict + ranking order", check_reranker)

# ─────────────────────────────────────────
# 7. FULL END-TO-END RAG QUERY
# ─────────────────────────────────────────
print(f"\n{INFO} [7/7] Full end-to-end RAG query...")

from langchain_community.llms import Ollama

def check_full_rag():
    query = "What is machine learning?"

    # Embed + retrieve
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db  = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=emb)
    vector_docs = db.similarity_search(query, k=4)

    # BM25
    stored    = db.get()
    texts     = stored["documents"]
    metas     = stored.get("metadatas") or [{}] * len(texts)
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(re.findall(r"\w+", query.lower()))
    top_idx   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]

    from langchain_core.documents import Document as LCDoc
    keyword_docs = [LCDoc(page_content=texts[i], metadata=metas[i]) for i in top_idx]

    # Merge + deduplicate
    seen, combined = set(), []
    for doc in (vector_docs + keyword_docs):
        if doc.page_content not in seen:
            combined.append(doc); seen.add(doc.page_content)
    combined = combined[:4]

    # Rerank
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs    = [(query, d.page_content) for d in combined]
    rscores  = reranker.predict(pairs)
    ranked   = [doc for _, doc in sorted(zip(rscores, combined), key=lambda x: x[0], reverse=True)]

    context  = "\n\n".join(d.page_content for d in ranked)

    prompt = f"""You are a document assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {query}
Answer:"""

    llm    = Ollama(model="phi3")
    t0     = time.time()
    answer = llm.invoke(prompt)
    elapsed = time.time() - t0

    if not answer or len(answer.strip()) < 10:
        raise Exception(f"Answer too short or empty: '{answer}'")

    preview = answer.strip()[:200].replace("\n", " ")
    return f"Answer ({elapsed:.1f}s): '{preview}...'"

check("Full RAG pipeline (retrieve+rerank+LLM)", check_full_rag)

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
