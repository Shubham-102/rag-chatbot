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

## Demo
Upload any PDF → ask questions → get answers with page citations.

## Features
- Upload any PDF and chat with it instantly
- Answers grounded in your documents — cites source pages
- Conversation memory across messages
- Powered by Groq (Llama 3.1) — free and fast
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