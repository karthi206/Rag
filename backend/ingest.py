"""
ingest.py — Production-grade PDF ingestion pipeline for RAG.

Usage:
    python ingest.py

Loads all PDFs from the `documents/` folder, splits them into chunks,
embeds them with SentenceTransformer, and stores in a persistent Chroma vectorstore.
Idempotent: if vectorstore already exists, new chunks are added without rebuilding.
"""

import os
import sys
from dotenv import load_dotenv

# Load HF_TOKEN and other env variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import SentenceTransformerEmbeddings


# -----------------------------
# Config (can be overridden via .env)
# -----------------------------
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents")
VECTORSTORE_DIR  = os.getenv("VECTORSTORE_DIR", "vectorstore")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_MODEL      = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


# -----------------------------
# Step 1: Guard — check documents folder
# -----------------------------
if not os.path.isdir(DOCUMENTS_FOLDER):
    print(f"[ERROR] Documents folder not found: '{DOCUMENTS_FOLDER}'")
    sys.exit(1)

pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print(f"[ERROR] No PDF files found in '{DOCUMENTS_FOLDER}'. Aborting.")
    sys.exit(1)

print(f"[INFO] Found {len(pdf_files)} PDF file(s) in '{DOCUMENTS_FOLDER}'")


# -----------------------------
# Step 2: Load all PDFs (with per-file error handling)
# -----------------------------
documents = []
skipped   = []

for file in pdf_files:
    file_path = os.path.join(DOCUMENTS_FOLDER, file)
    try:
        loader = PyPDFLoader(file_path)
        docs   = loader.load()

        if not docs:
            print(f"[WARN] '{file}' loaded 0 pages — skipping.")
            skipped.append(file)
            continue

        # Attach filename as source metadata
        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)
        print(f"[OK]   Loaded '{file}' — {len(docs)} page(s)")

    except Exception as e:
        print(f"[ERROR] Failed to load '{file}': {e} — skipping.")
        skipped.append(file)

print(f"\n[INFO] Total pages loaded: {len(documents)}")
if skipped:
    print(f"[WARN] Skipped files: {skipped}")

if not documents:
    print("[ERROR] No documents were successfully loaded. Aborting.")
    sys.exit(1)


# -----------------------------
# Step 3: Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

splits = splitter.split_documents(documents)
print(f"[INFO] Created {len(splits)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

if len(splits) == 0:
    print("[ERROR] No chunks created. Check your documents and chunk settings.")
    sys.exit(1)


# -----------------------------
# Step 4: Load embedding model
# -----------------------------
print(f"[INFO] Loading embedding model: {EMBED_MODEL}")
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)


# -----------------------------
# Step 5: Idempotent vectorstore creation
# If vectorstore already exists, ADD new chunks rather than rebuilding.
# This prevents duplicate embeddings on repeated runs.
# -----------------------------
if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
    print(f"[INFO] Existing vectorstore found at '{VECTORSTORE_DIR}'. Adding new chunks...")
    db = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    db.add_documents(splits)
    print(f"[OK]   Added {len(splits)} chunks to existing vectorstore.")
else:
    print(f"[INFO] Creating new vectorstore at '{VECTORSTORE_DIR}'...")
    db = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    print(f"[OK]   New vectorstore created with {len(splits)} chunks.")


# -----------------------------
# Summary
# -----------------------------
total_in_db = len(db.get()["documents"])
print(f"\n{'='*50}")
print(f"  INGESTION COMPLETE")
print(f"{'='*50}")
print(f"  Files loaded     : {len(pdf_files) - len(skipped)}")
print(f"  Files skipped    : {len(skipped)}")
print(f"  Chunks this run  : {len(splits)}")
print(f"  Total in DB      : {total_in_db}")
print(f"  Vectorstore      : {VECTORSTORE_DIR}")
print(f"{'='*50}")