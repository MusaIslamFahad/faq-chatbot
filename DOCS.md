# 💬 VisionChat FAQ — Full Documentation

> **CodeAlpha AI Internship · Task 2 — Chatbot for FAQs**  
> Built by **Md. Musa Islam Fahad** · CSE (Data Science) · Daffodil International University

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-3.0-orange)](https://sbert.net)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

**VisionChat FAQ** is a production-grade semantic FAQ chatbot that goes far beyond
basic keyword matching. It understands the *meaning* of your question — not just
the words — using state-of-the-art sentence embeddings and vector search.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🧠 Semantic search | `all-MiniLM-L6-v2` sentence embeddings (not TF-IDF keyword matching) |
| 🗂️ Vector database | ChromaDB — persistent, local, production-ready |
| 🎯 Intent classification | 8 intents: greeting, farewell, gratitude, complaint, pricing, account, billing, technical |
| 📝 NLP preprocessing | NLTK + SpaCy — tokenisation, stopword removal, lemmatisation, NER |
| 📊 Analytics dashboard | Intent distribution, category breakdown, unanswered queries, response time |
| 👍 Feedback system | Per-response helpful/not-helpful rating stored in SQLite |
| 🔌 REST API | FastAPI backend with Swagger docs at `/docs` |
| 📖 FAQ browser | Searchable, filterable FAQ database UI |
| 🌐 Free deployment | Hugging Face Spaces (CPU) or Streamlit Cloud — permanent public URL |
| 🐳 Docker | Full Dockerfile for containerised deployment |
| 📈 Confidence gating | High / low / no-match routing with appropriate messaging |

---

## 🗂️ Project Structure

```
faq_chatbot/
│
├── app.py                      ← Streamlit main application (4 tabs)
├── api.py                      ← FastAPI REST backend
│
├── src/
│   ├── preprocessor.py         ← NLP pipeline (NLTK + SpaCy)
│   ├── embedder.py             ← Sentence Transformers + ChromaDB engine
│   ├── intent_classifier.py    ← Intent detection (embedding-based)
│   └── chatbot.py              ← Core orchestrator (FAQChatbot class)
│
├── utils/
│   └── analytics_db.py         ← SQLite analytics logger
│
├── data/
│   ├── faqs.json               ← FAQ dataset (40+ entries, 7 categories)
│   └── chroma_db/              ← ChromaDB vector index (auto-created)
│
├── logs/
│   └── analytics.db            ← SQLite query log (auto-created)
│
├── .streamlit/
│   └── config.toml             ← Dark theme + server config
│
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── DOCS.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/MusaIslamFahad/CodeAlpha_FAQChatbot
cd CodeAlpha_FAQChatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP models (first time only)
```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','stopwords','wordnet','averaged_perceptron_tagger']]"
python -m spacy download en_core_web_sm
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. (Optional) Run the FastAPI backend
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
```

---

## 🔌 REST API Reference

Once the FastAPI server is running, visit `http://localhost:8000/docs` for interactive Swagger UI.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Main chat endpoint |
| `GET` | `/faqs` | List all FAQs |
| `GET` | `/faqs/{category}` | FAQs by category |
| `GET` | `/categories` | List all categories |
| `POST` | `/feedback` | Submit helpful/not helpful rating |
| `GET` | `/analytics/summary` | Session analytics |
| `GET` | `/analytics/queries` | Recent query log |
| `GET` | `/analytics/unanswered` | Unmatched queries |
| `GET` | `/health` | Health check |

**Example chat request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?", "session_id": "abc123"}'
```

---

## 🧠 Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  NLP Preprocessor   │  ← lowercase, expand contractions, tokenise,
│  (NLTK + SpaCy)     │    remove stopwords, lemmatise (NLTK WordNet)
└──────────┬──────────┘    optional NER via SpaCy en_core_web_sm
           │
           ▼
┌─────────────────────┐
│ Intent Classifier   │  ← encodes query via all-MiniLM-L6-v2
│ (embedding-based)   │    cosine sim over 8 intent exemplar centroids
└──────────┬──────────┘
           │
     ┌─────┴──────────────────────────────┐
     │                                    │
 greeting/farewell               FAQ intent / general
 gratitude/complaint                      │
     │                                    ▼
 template response          ┌─────────────────────────┐
 human handoff              │ Sentence Transformer     │
                            │ all-MiniLM-L6-v2         │
                            │ encodes user query → 384d│
                            └──────────┬──────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────┐
                            │  ChromaDB               │
                            │  cosine similarity      │
                            │  over FAQ embeddings    │
                            └──────────┬──────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────┐
                            │ Confidence gating        │
                            │ ≥65% → confident answer  │
                            │ 35-65% → low-conf warn   │
                            │ <35%  → no-match msg     │
                            └──────────┬──────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────┐
                            │ Structured ChatResponse  │
                            │ + alternative questions  │
                            │ + category + metadata    │
                            └──────────┬──────────────┘
                                       │
                            ┌──────────┴──────────┐
                            │                     │
                     SQLite Logger         Streamlit UI
                     (every query)        chat bubbles
                                          confidence bar
                                          feedback buttons
```

---

## 📦 Adding Your Own FAQs

Edit `data/faqs.json`. Each FAQ entry follows this schema:

```json
{
  "id": "unique_id",
  "category": "Category Name",
  "question": "Your question here?",
  "answer": "Your detailed answer here."
}
```

Delete `data/chroma_db/` after adding FAQs to force a re-index on next startup.

---

## ☁️ Deployment

### Hugging Face Spaces (Recommended — Free, CPU, permanent URL)
1. Create a Space at huggingface.co/spaces → choose Streamlit SDK
2. Push this repo to the Space:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/visionchat-faq
git push hf main
```

### Streamlit Cloud (Free, simple)
1. Push repo to GitHub
2. Go to share.streamlit.io → connect repo → select `app.py` → Deploy

### Docker
```bash
docker build -t visionchat-faq .
docker run -p 8501:8501 visionchat-faq
```

---

## 🤝 Author

**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University, Dhaka  
📧 musa.islam.fahad@gmail.com  
🌐 [Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad) · [LinkedIn](https://linkedin.com/in/md-musa-islam-fahad-b18759249)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
