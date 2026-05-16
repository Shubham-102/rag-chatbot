import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())
    print(f"Total pages loaded: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def embed_and_store(chunks):
    print("Loading embedding model on your GPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},  # uses your RTX 3070
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Done! Stored in: {CHROMA_DIR}/")
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)
    print("\nIngestion complete. Ready to chat!")