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
## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 5. Add your documents
Place any PDF files in the `data/` folder, then run:
```bash
python ingest.py
```

### 6. Run the app
```bash
python -m streamlit run app.py
```

## Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — completely free.