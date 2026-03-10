import os
import re
import tempfile
import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Cache heavy models
# -----------------------------
@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_llm():
    return Ollama(model="phi3")


embeddings = load_embeddings()
reranker = load_reranker()
llm = load_llm()


# -----------------------------
# Initialize chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("📄 Advanced RAG Document Assistant")


# -----------------------------
# Display conversation history
# -----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# -----------------------------
# Upload PDFs
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_documents = []

    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        loader = PyPDFLoader(temp_file.name)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_documents.extend(docs)

    st.success(f"{len(uploaded_files)} document(s) loaded successfully!")


    # -----------------------------
    # Chunk documents
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    splits = splitter.split_documents(all_documents)
    st.write("Total chunks created:", len(splits))


    # -----------------------------
    # Build BM25 index
    # -----------------------------
    chunk_texts = [doc.page_content for doc in splits]
    tokenized_corpus = [re.findall(r"\w+", text.lower()) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)


    # -----------------------------
    # Build vector database
    # -----------------------------
    db = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory="vectorstore"
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})


    # -----------------------------
    # Hybrid search
    # -----------------------------
    def hybrid_search(query, k=4):

        vector_results = retriever.invoke(query)

        tokenized_query = re.findall(r"\w+", query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        top_keyword_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]

        keyword_results = [splits[i] for i in top_keyword_indices]

        combined = vector_results + keyword_results

        seen = set()
        unique_docs = []

        for doc in combined:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs[:k]


    # -----------------------------
    # Re-ranking
    # -----------------------------
    def rerank_documents(query, docs, top_k=4):

        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        ranked_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for _, doc in ranked_docs[:top_k]]


    # -----------------------------
    # Chat input
    # -----------------------------
    query = st.chat_input("Ask a question about the uploaded documents")

    if query:

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.chat_history.append(("user", query))


        docs = hybrid_search(query)
        docs = rerank_documents(query, docs)


        context = "\n\n".join(doc.page_content for doc in docs)


        sources = set()

        for doc in docs:
            page = doc.metadata.get("page", None)
            source = doc.metadata.get("source", "document")

            if page is not None:
                sources.add(f"{source} - Page {page + 1}")


        history_text = ""
        for role, message in st.session_state.chat_history:
            history_text += f"{role}: {message}\n"


        prompt = f"""
You are a document assistant.

Answer ONLY using the provided document context.

If the answer cannot be found in the document say:
"I cannot find the answer in the provided documents."

Conversation history:
{history_text}

Document context:
{context}

Question:
{query}

Answer:
"""


        with st.chat_message("assistant"):

            placeholder = st.empty()
            answer = ""

            response = llm.stream(prompt)

            for chunk in response:
                answer += chunk
                placeholder.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))


        st.markdown("### Sources")

        for src in sources:
            clean_name = os.path.basename(src)
            st.write(clean_name)