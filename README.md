---
title: RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: false
---

# RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in your own documents — no hallucination, with source citations.

## Live Demo
🚀 [Try it live on Hugging Face Spaces](https://huggingface.co/spaces/shubhaM-Maheshwari/rag-chatbot)

## Features
- Upload any PDF and chat with it instantly
- Answers grounded in your documents — cites source pages
- Conversation memory across messages
- Powered by Groq Llama 3.1 — free and fast
- Local embeddings via HuggingFace sentence-transformers

## Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM | Groq (Llama 3.1 8B) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| UI | Streamlit |

## Architecture

    PDF → Chunking (500 tokens) → Embeddings → ChromaDB
                                                    ↓
    User Question → Embed → Similarity Search → Top 4 Chunks
                                                    ↓
                                            Groq LLM → Answer + Page Sources

## Project Structure

    rag-chatbot/
    ├── app.py            # Streamlit UI
    ├── ingest.py         # PDF loader and ChromaDB ingestion
    ├── rag_chain.py      # RAG chain logic
    ├── requirements.txt
    ├── .env.example
    └── data/             # Place your PDFs here

## Run Locally

**1. Clone the repo**

    git clone https://github.com/Shubham-102/rag-chatbot.git
    cd rag-chatbot

**2. Create virtual environment**

    python -m venv venv
    venv\Scripts\activate     # Windows
    source venv/bin/activate  # Mac/Linux

**3. Install dependencies**

    pip install -r requirements.txt

**4. Set up environment variables**

    cp .env.example .env

Add your GROQ_API_KEY to .env — get a free key at https://console.groq.com

**5. Ingest your documents**

    python ingest.py

**6. Run the app**

    python -m streamlit run app.py