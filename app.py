"""
app.py — Production-grade RAG Document Assistant (Streamlit)

Features:
  - PDF upload with deduplication guard (no re-processing on Streamlit reruns)
  - Hybrid search: BM25 + Chroma vector retrieval
  - Cross-encoder re-ranking
  - Streaming LLM responses (token-by-token)
  - Rolling chat history (last N turns)
  - Graceful error handling (corrupt PDFs, Ollama offline)
  - Sidebar: document list, model info, clear-chat button
  - BM25 bootstrapped from disk on startup (chat works without re-uploading)
"""

import os
import re
import tempfile
import streamlit as st

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────
# Config (overridable via .env)
# ─────────────────────────────────────────
VECTORSTORE_DIR    = os.getenv("VECTORSTORE_DIR", "vectorstore")
MODEL_NAME         = os.getenv("MODEL_NAME", "phi3")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS", "5"))
K_RETRIEVE         = int(os.getenv("K_RETRIEVE", "4"))
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "50"))


# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    #page_icon="📚",
    layout="wide"
)


# ─────────────────────────────────────────
# Cache heavy models (loaded once per session)
# ─────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_llm():
    return Ollama(model=MODEL_NAME)


embeddings = load_embeddings()
reranker   = load_reranker()
llm        = load_llm()


# ─────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────
defaults = {
    "chat_history":    [],
    "bm25":            None,
    "splits":          [],
    "db":              None,
    "processed_files": set(),   # tracks filenames already ingested this session
    "uploaded_names":  [],      # display list for sidebar
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────
# Bootstrap: load existing vectorstore + BM25 on startup
# So chat works even without uploading in this session
# ─────────────────────────────────────────
if st.session_state.db is None:
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        try:
            st.session_state.db = Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=embeddings
            )
            # Pull stored docs to rebuild BM25 index
            stored = st.session_state.db.get()
            if stored and stored.get("documents"):
                docs_text = stored["documents"]
                tokenized = [re.findall(r"\w+", t.lower()) for t in docs_text]
                st.session_state.bm25 = BM25Okapi(tokenized)
                # Rebuild splits list for BM25 index alignment
                from langchain_core.documents import Document as LCDoc
                metas = stored.get("metadatas") or [{}] * len(docs_text)
                st.session_state.splits = [
                    LCDoc(page_content=t, metadata=m)
                    for t, m in zip(docs_text, metas)
                ]
        except Exception as e:
            st.warning(f"Could not load existing vectorstore: {e}")


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.divider()

    st.subheader("🤖 Model Info")
    st.write(f"**LLM:** `{MODEL_NAME}` (Ollama)")
    st.write(f"**Embeddings:** `{EMBED_MODEL}`")
    st.write(f"**Reranker:** `ms-marco-MiniLM-L-6-v2`")
    st.write(f"**Chunk size:** {CHUNK_SIZE} | **Overlap:** {CHUNK_OVERLAP}")
    st.write(f"**Top-K retrieve:** {K_RETRIEVE}")
    st.divider()

    st.subheader("📄 Uploaded Documents")
    if st.session_state.uploaded_names:
        for name in st.session_state.uploaded_names:
            st.write(f"• {name}")
    elif st.session_state.db is not None:
        st.write("*(loaded from disk)*")
    else:
        st.write("*No documents yet.*")
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────
st.title("Advanced RAG Document Assistant")
st.caption(f"Hybrid search · Cross-encoder reranking · Streaming · Ollama `{MODEL_NAME}`")
st.divider()


# ─────────────────────────────────────────
# Display conversation history
# ─────────────────────────────────────────
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# ─────────────────────────────────────────
# Upload PDFs
# ─────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    # FIXED: Only process files that haven't been ingested in this session.
    # Without this guard, Streamlit's re-run on every interaction would
    # re-embed all files on every user action, causing duplicate chunks.
    new_files = [
        f for f in uploaded_files
        if f.name not in st.session_state.processed_files
    ]

    if new_files:
        all_documents = []
        failed_files  = []

        for uploaded_file in new_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_path = temp_file.name
            try:
                temp_file.write(uploaded_file.read())
                temp_file.close()

                loader = PyPDFLoader(temp_path)
                docs   = loader.load()

                if not docs:
                    st.warning(f"⚠️ '{uploaded_file.name}' appears empty — skipped.")
                    failed_files.append(uploaded_file.name)
                    continue

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(docs)
                st.session_state.processed_files.add(uploaded_file.name)
                st.session_state.uploaded_names.append(uploaded_file.name)

            except Exception as e:
                st.error(f"❌ Could not load '{uploaded_file.name}': {e}")
                failed_files.append(uploaded_file.name)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        if all_documents:
            # Chunk
            splitter   = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            new_splits = splitter.split_documents(all_documents)

            # Vectorstore (idempotent)
            if st.session_state.db is None:
                if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
                    st.session_state.db = Chroma(
                        persist_directory=VECTORSTORE_DIR,
                        embedding_function=embeddings
                    )
                    st.session_state.db.add_documents(new_splits)
                else:
                    st.session_state.db = Chroma.from_documents(
                        new_splits,
                        embeddings,
                        persist_directory=VECTORSTORE_DIR
                    )
            else:
                st.session_state.db.add_documents(new_splits)

            # Accumulate splits & rebuild BM25
            st.session_state.splits.extend(new_splits)
            chunk_texts      = [doc.page_content for doc in st.session_state.splits]
            tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in chunk_texts]
            st.session_state.bm25 = BM25Okapi(tokenized_corpus)

            loaded_count = len(new_files) - len(failed_files)
            st.success(
                f"✅ {loaded_count} document(s) ingested — "
                f"{len(all_documents)} page(s) — "
                f"{len(new_splits)} chunk(s) created."
            )


# ─────────────────────────────────────────
# Gate: require documents before showing chat
# ─────────────────────────────────────────
if st.session_state.db is None:
    st.info("👆 Upload one or more PDFs above to get started.")
    st.stop()

retriever = st.session_state.db.as_retriever(search_kwargs={"k": K_RETRIEVE})
bm25      = st.session_state.bm25
splits    = st.session_state.splits


# ─────────────────────────────────────────
# Hybrid search
# ─────────────────────────────────────────
def hybrid_search(query: str, k: int = K_RETRIEVE) -> list:
    vector_results   = retriever.invoke(query)
    tokenized_query  = re.findall(r"\w+", query.lower())

    if not tokenized_query or bm25 is None:
        return vector_results[:k]

    bm25_scores          = bm25.get_scores(tokenized_query)
    top_keyword_indices  = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    keyword_results = [splits[i] for i in top_keyword_indices if i < len(splits)]

    combined   = vector_results + keyword_results
    seen       = set()
    unique_docs = []
    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:k]


# ─────────────────────────────────────────
# Re-ranking
# ─────────────────────────────────────────
def rerank_documents(query: str, docs: list, top_k: int = K_RETRIEVE) -> list:
    if not docs:
        return []
    pairs      = [(query, doc.page_content) for doc in docs]
    scores     = reranker.predict(pairs)
    ranked     = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ─────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────
query = st.chat_input("Ask a question about your uploaded documents…")

if query:
    if not query.strip():
        st.warning("Please enter a non-empty question.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.chat_history.append(("user", query))

    # Retrieve + rerank
    docs = hybrid_search(query)
    docs = rerank_documents(query, docs)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Sources
    sources = set()
    for doc in docs:
        page   = doc.metadata.get("page")
        source = doc.metadata.get("source", "document")
        if page is not None:
            sources.add(f"📄 {source} — Page {int(page) + 1}")

    # Rolling chat history (last N turns)
    recent_history = st.session_state.chat_history[-(MAX_HISTORY_TURNS * 2):]
    history_text   = "".join(f"{r}: {m}\n" for r, m in recent_history)

    prompt = f"""You are a precise document assistant. Your job is to answer questions based ONLY on the provided document context.

Rules:
1. Answer ONLY from the document context below.
2. If the answer is not in the documents, say: "I cannot find the answer in the provided documents."
3. Be concise and factual.
4. Do not make up information.

Conversation history (last {MAX_HISTORY_TURNS} turns):
{history_text}
Document context:
{context}

Question: {query}

Answer:"""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer      = ""

        try:
            # FIXED: Removed st.spinner() wrapper so tokens display live (true streaming)
            for chunk in llm.stream(prompt):
                answer += chunk
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)

        except Exception as e:
            answer = f"⚠️ **LLM Error:** Could not get a response. Is Ollama running?\n\n`{e}`"
            placeholder.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))

    # Sources section
    if sources:
        st.divider()
        st.markdown("**📌 Sources**")
        for src in sorted(sources):
            st.write(src)
    else:
        st.caption("*No specific page sources identified.*")