from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

loader = PyPDFLoader("documents/AI.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

splits = splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    splits,
    embeddings,
    persist_directory="vectorstore"
)

db.persist()

print("Embeddings stored successfully.")