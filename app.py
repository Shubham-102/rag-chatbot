import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore, embeddings

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(question, chat_history, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer using only the context below.
If the answer is not in the context, say 'I do not have enough information to answer that.'

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    docs = retriever.invoke(question)
    context = format_docs(docs)

    messages = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        question=question
    )

    response = llm.invoke(messages)

    pages = []
    if docs:
        pages = sorted(set([
            doc.metadata.get("page", 0) + 1
            for doc in docs
            if isinstance(doc.metadata.get("page"), int)
        ]))

    return response.content, pages

def ingest_uploaded_file(uploaded_file, embeddings):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    os.unlink(tmp_path)
    return len(chunks), vectorstore

st.title("RAG Chatbot")
st.caption("Ask questions about your documents")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Processing..."):
                vectorstore, embeddings = load_retriever()
                chunks, vectorstore = ingest_uploaded_file(
                    uploaded_file, embeddings
                )
                st.success(f"Done! Added {chunks} chunks.")
                st.cache_resource.clear()

    st.divider()
    st.markdown("**Documents loaded:**")
    if os.path.exists(CHROMA_DIR):
        st.success("ChromaDB ready")
    else:
        st.error("No documents found. Run ingest.py first.")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: pages {message['sources']}")

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vectorstore, _ = load_retriever()
            answer, pages = get_answer(
                question,
                st.session_state.chat_history,
                vectorstore
            )
        st.markdown(answer)
        if pages:
            st.caption(f"Sources: pages {pages}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": pages
    })

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))