---
title: AI Research Assistant
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# 🔬 AI Research Assistant

A **production-grade RAG (Retrieval-Augmented Generation)** system that lets you chat with your documents — PDFs, DOCX files, web pages — using any LLM.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI  ──────────────────────────────────────────────────
    │                                                           │
    ▼                                                           ▼
RAG Engine                                            Document Ingestion
    │                                                   PDF / DOCX / Web
    ├── Retriever ──► ChromaDB Vector Store ◄─────── Embeddings
    │                (semantic similarity)         (local / OpenAI)
    │
    ├── Context Builder
    │     (format, deduplicate, rank)
    │
    ├── Prompt Builder
    │     (system + chat history + context + question)
    │
    └── LLM (OpenAI GPT-4o / Groq Llama 3) ──► Streaming Answer
```

## 📁 Project Structure

```
ai_research_assistant/
├── app.py                  ← Streamlit UI (main entry point)
├── requirements.txt        ← All dependencies
├── .env.example            ← Config template → copy to .env
│
├── src/
│   ├── config.py           ← Settings from .env
│   ├── logger.py           ← Loguru logging setup
│   ├── llm.py              ← LLM factory (OpenAI / Groq)
│   ├── embeddings.py       ← Embedding model factory
│   ├── ingestion.py        ← PDF / DOCX / Web → LangChain Docs
│   ├── vectorstore.py      ← ChromaDB manager
│   ├── rag_chain.py        ← Core RAG pipeline + memory
│   └── utils.py            ← Helper functions
│
├── vectorstore/            ← ChromaDB persisted data (auto-created)
├── uploads/                ← Temp upload dir (auto-created)
└── logs/                   ← Log files (auto-created)
```

---

## ⚡ Quick Start

### Step 1 — Prerequisites
- Python 3.9+
- Git

### Step 2 — Clone & Setup
```bash
pip install -r requirements.txt
```

### Step 3 — Configure API Key
Copy `.env.example` to `.env` and set your key:

**Option A: Groq (FREE, fast)**
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```
Get free key at → https://console.groq.com

**Option B: OpenAI**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your_key_here
```

### Step 4 — Run
```bash
streamlit run app.py
```

### Step 5 — Open in browser
Visit → **http://localhost:8501**

---

## 🚀 Features

| Feature | Details |
|---------|---------|
| **PDF Ingestion** | PyMuPDF (primary) + pypdf (fallback) |
| **DOCX Ingestion** | python-docx |
| **Web Scraping** | trafilatura + BeautifulSoup fallback |
| **Embeddings** | sentence-transformers (local, free) OR OpenAI |
| **Vector Store** | ChromaDB (persistent, on-disk) |
| **LLM** | Groq Llama 3.3 (free) OR OpenAI GPT-4o |
| **Streaming** | Token-by-token streaming responses |
| **Memory** | Sliding-window conversation history (6 turns) |
| **Deduplication** | MD5 hash-based chunk deduplication |
| **Source Citation** | Auto-extracted source pills in UI |
| **Follow-up Q** | AI-generated follow-up question suggestions |
| **Summarisation** | Full-document summarisation per source |
| **Library View** | Manage, view, delete ingested documents |
| **Source Filter** | Ask questions about one specific document |

---

## 🔧 Configuration Options (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RESULTS` | `6` | Chunks retrieved per query |
| `TEMPERATURE` | `0.2` | LLM creativity (0=factual, 1=creative) |

---

## 📝 Example Queries

Once you've uploaded a research paper:
- *"What is the main contribution of this paper?"*
- *"Explain the methodology used in section 3"*
- *"What datasets were used for evaluation?"*
- *"Compare the results with previous work"*
- *"What are the limitations mentioned?"*

---

## 🛠️ Troubleshooting

**"Module not found" error**
→ Make sure all requirements are installed

**ChromaDB error on first run**
→ Delete `./vectorstore` folder and restart

**Slow embeddings on first run**
→ Model is downloading (~90MB). Next run will be instant.

**URL scraping fails**
→ Some sites block bots. Try a different URL or use the PDF download.