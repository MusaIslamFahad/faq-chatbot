<!--
---
title: VisionChat FAQ
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
short_description: Semantic FAQ Chatbot · Sentence Transformers + ChromaDB
---

-->

<div align="center">
  
# 💬 VisionChat FAQ Assistant

Semantic FAQ chatbot using **Sentence Transformers + ChromaDB**.  
Built by **Md. Musa Islam Fahad** · CodeAlpha AI Internship Task 2.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-3.0-orange?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-blue?style=for-the-badge)
![SpaCy](https://img.shields.io/badge/SpaCy-3.7-09A3D5?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Analytics-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Internship](https://img.shields.io/badge/CodeAlpha-AI%20Internship%20·%20Task%202-orange?style=for-the-badge)

![VisionChat FAQ Banner](https://raw.githubusercontent.com/MusaIslamFahad/faq-chatbot/main/assets/banner.png)

**Production-grade semantic FAQ chatbot - Sentence Transformers + ChromaDB vector search, 8-intent classifier, FastAPI REST backend, Streamlit dashboard, and SQLite analytics.**

> CodeAlpha AI Internship · Task 2  
> Built by **Md. Musa Islam Fahad** · CSE (Data Science) · Daffodil International University

</div>

---

## 📖 Overview

**VisionChat FAQ** is a production-grade semantic FAQ chatbot built as part of the [CodeAlpha](https://www.codealpha.tech/) AI/ML Internship. Unlike keyword-based systems, it uses `all-MiniLM-L6-v2` **Sentence Transformers** to understand the *meaning* behind a question and retrieves the best-matching FAQ via **ChromaDB** cosine vector search.

An **NLP preprocessing pipeline** (NLTK + SpaCy) cleans and lemmatises each query before it reaches an **embedding-based intent classifier** that routes greetings, complaints, and out-of-scope queries to appropriate template responses and passes genuine FAQ intents to the vector search engine.

An interactive **Streamlit dashboard** provides a four-tab UI (Chat · FAQ Browser · Analytics · About), while a **FastAPI REST backend** exposes the same capability via a fully documented API. Every query is logged to **SQLite** for session analytics, unanswered query tracking, and per-response feedback.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Semantic Search** | `all-MiniLM-L6-v2` sentence embeddings - understands meaning, not just keywords |
| 🗂️ **Vector Database** | ChromaDB - persistent, local, production-ready |
| 🎯 **Intent Classification** | 8 intents: greeting, farewell, gratitude, complaint, pricing, account, billing, technical |
| 📝 **NLP Preprocessing** | NLTK + SpaCy - tokenisation, stopword removal, lemmatisation, NER |
| 📊 **Analytics Dashboard** | Intent distribution, category breakdown, unanswered queries, response time charts |
| 👍 **Feedback System** | Per-response helpful / not-helpful rating stored in SQLite |
| 🔌 **REST API** | FastAPI backend with full Swagger docs at `/docs` |
| 📖 **FAQ Browser** | Searchable, filterable FAQ database UI with direct "Ask this →" action |
| 📈 **Confidence Gating** | High (≥65%) / low (35–64%) / no-match (<35%) routing with appropriate messaging |
| 🐳 **Docker Ready** | Full Dockerfile for containerised CPU deployment |
| ☁️ **Multi-platform Deploy** | Hugging Face Spaces · Streamlit Cloud · Docker |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Embedding Model | [Sentence Transformers](https://sbert.net/) `all-MiniLM-L6-v2` |
| Vector Database | [ChromaDB](https://www.trychroma.com/) 0.5 (persistent, local) |
| NLP Pipeline | [NLTK](https://www.nltk.org/) 3.8 + [SpaCy](https://spacy.io/) 3.7 `en_core_web_sm` |
| REST Backend | [FastAPI](https://fastapi.tiangolo.com/) 0.111 + Uvicorn |
| UI / Dashboard | [Streamlit](https://streamlit.io/) 1.35 |
| Analytics Charts | [Plotly](https://plotly.com/python/) 5.22 |
| Analytics Store | SQLite (via `sqlite-utils`) |
| Deep Learning | PyTorch 2.0+ |

---

## 📁 Project Structure

```
faq_chatbot/
│
├── app.py                      # Streamlit main application (4 tabs) - entry point
├── api.py                      # FastAPI REST backend
│
├── src/
│   ├── preprocessor.py         # NLP pipeline (NLTK + SpaCy)
│   ├── embedder.py             # Sentence Transformers + ChromaDB engine
│   ├── intent_classifier.py    # Intent detection (embedding-based, 8 intents)
│   └── chatbot.py              # Core orchestrator - FAQChatbot class
│
├── utils/
│   └── analytics_db.py         # SQLite analytics logger
│
├── data/
│   ├── faqs.json               # FAQ dataset (40+ entries, 7 categories)
│   └── chroma_db/              # ChromaDB vector index (auto-created on first run)
│
├── logs/
│   └── analytics.db            # SQLite query log (auto-created)
│
├── .streamlit/
│   └── config.toml             # Dark theme + server configuration
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build definition
├── .env.example                # Environment variable template
├── .gitignore
├── DOCS.md                     # Full technical documentation
└── README.md
```

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/MusaIslamFahad/faq-chatbot.git
cd faq-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLP models *(first time only)*

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']]"
python -m spacy download en_core_web_sm
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
# Open http://localhost:8501
```

### 6. *(Optional)* Run the FastAPI backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI: http://localhost:8000/docs
```

---

## 🚀 Usage

### Streamlit Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The dashboard has four tabs:

| Tab | What you can do |
|-----|-----------------|
| 🗨️ **Chat** | Ask questions, view confidence score, click alternative suggestions, give feedback |
| 📖 **FAQ Browser** | Search and filter the full FAQ dataset by category or keyword |
| 📊 **Analytics** | View KPIs, intent distribution chart, category breakdown, recent queries, unanswered log |
| ℹ️ **About** | Full architecture overview and tech stack reference |

The **sidebar** lets you browse FAQs by category, click example questions, and clear the conversation.

---

## 🔌 REST API Reference

Start the FastAPI server and visit `http://localhost:8000/docs` for the interactive Swagger UI.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Main chat endpoint |
| `GET` | `/faqs` | List all FAQs |
| `GET` | `/faqs/{category}` | FAQs filtered by category |
| `GET` | `/categories` | List all available categories |
| `POST` | `/feedback` | Submit helpful / not-helpful rating |
| `GET` | `/analytics/summary` | Session-level analytics summary |
| `GET` | `/analytics/queries` | Recent query log (last 50) |
| `GET` | `/analytics/unanswered` | Queries with no confident match |
| `GET` | `/health` | Health check |

**Example request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?", "session_id": "abc123"}'
```

**Example response:**

```json
{
  "answer": "You can reset your password from the login page by clicking 'Forgot Password'...",
  "confidence": 0.87,
  "category": "Account",
  "matched_question": "How do I reset my password?",
  "match_type": "semantic",
  "alternative_questions": ["How do I change my email?", "I can't log in to my account"],
  "response_time_ms": 42,
  "requires_human": false
}
```

---

## 🧠 Architecture

```
User Query
      │
      ▼
┌─────────────────────┐
│  NLP Preprocessor   │  ← lowercase, expand contractions, tokenise,
│  (NLTK + SpaCy)     │    remove stopwords, lemmatise, optional NER
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Intent Classifier   │  ← encodes query via all-MiniLM-L6-v2
│ (embedding-based)   │    cosine sim over 8 intent exemplar centroids
└──────────┬──────────┘
           │
    ┌──────┴─────────────────────────────┐
    │                                    │
greeting / farewell / gratitude     FAQ intent / general
complaint / out-of-scope                 │
    │                                    ▼
template response /          ┌───────────────────────────┐
human handoff                │  Sentence Transformer     │
                             │  all-MiniLM-L6-v2         │
                             │  encodes query → 384d vec │
                             └──────────┬────────────────┘
                                        │
                                        ▼
                             ┌───────────────────────────┐
                             │  ChromaDB                 │
                             │  cosine similarity search │
                             │  over persistent FAQ index│
                             └──────────┬────────────────┘
                                        │
                                        ▼
                             ┌───────────────────────────┐
                             │  Confidence Gating        │
                             │  ≥65%  → confident answer │
                             │  35–64% → low-conf warning│
                             │  <35%  → no-match message │
                             └──────────┬────────────────┘
                                        │
                                        ▼
                             ┌───────────────────────────┐
                             │  Structured ChatResponse  │
                             │  answer + alternatives    │
                             │  + category + metadata    │
                             └────────┬──────────────────┘
                                      │
                         ┌────────────┴──────────────┐
                         │                           │
                  SQLite Logger              Streamlit UI / FastAPI
                  (every query)             chat bubbles · confidence bar
                                            feedback buttons · analytics
```

**Step-by-step:**

1. **Preprocessing**: NLTK and SpaCy clean, tokenise, and lemmatise the raw query.
2. **Intent Classification**: The cleaned query is embedded and compared against 8 intent centroids. Non-FAQ intents receive template responses immediately.
3. **Vector Search**: For FAQ intents, the query embedding is matched against the ChromaDB index using cosine similarity, returning the top-3 most semantically similar FAQs.
4. **Confidence Gating**: Scores are thresholded to route high-confidence answers, low-confidence warnings, and no-match graceful deflections.
5. **Response Construction**: A structured `ChatResponse` is built with the answer, category, matched question, alternatives, and response time.
6. **Logging**: Every query, intent, match score, and response time is written to SQLite for analytics.
7. **UI / API**: Streamlit renders the chat bubble, confidence bar, and feedback buttons. FastAPI exposes identical logic over HTTP.

---

## 🎯 Intent Categories

| Intent | Description | Response Strategy |
|--------|-------------|-------------------|
| `greeting` | Hello, hi, hey | Template greeting |
| `farewell` | Bye, goodbye, see you | Template farewell |
| `gratitude` | Thank you, thanks | Template acknowledgement |
| `complaint` | Angry, frustrated, upset | Human handoff message |
| `pricing` | Cost, plan, price, subscription | FAQ vector search |
| `account` | Login, password, profile, signup | FAQ vector search |
| `billing` | Invoice, refund, charge, payment | FAQ vector search |
| `technical` | Error, bug, app, not working | FAQ vector search |

---

## 📈 Confidence Gating

| Score Range | Tier | Behaviour |
|-------------|------|-----------|
| ≥ 65% | **High confidence** | Returns the matched answer with full metadata |
| 35 – 64% | **Low confidence** | Returns the answer with a low-confidence disclaimer |
| < 35% | **No match** | Returns a graceful "I couldn't find an answer" message and logs the query as unanswered |

---

## 📦 Adding Your Own FAQs

Edit `data/faqs.json`. Each entry follows this schema:

```json
{
  "id": "unique_id",
  "category": "Category Name",
  "question": "Your question here?",
  "answer": "Your detailed answer here."
}
```

> **Important:** Delete `data/chroma_db/` after adding new FAQs to force a full re-index on the next startup.

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t visionchat-faq .

# Run on CPU
docker run -p 8501:8501 visionchat-faq

# Open http://localhost:8501
```

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** SDK
3. Push this repository to the Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/visionchat-faq
git push hf main
```

> The ChromaDB index and SQLite analytics database are auto-created on first startup — no manual setup required.

---

## ☁️ Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Select repo → `app.py` → Deploy

---

## 📋 Requirements

```
sentence-transformers==3.0.1
chromadb==0.5.3
fastapi==0.111.0
uvicorn[standard]==0.30.1
streamlit==1.35.0
nltk==3.8.1
spacy==3.7.5
scikit-learn==1.5.0
numpy==1.26.4
pandas==2.2.2
plotly==5.22.0
sqlite-utils==3.36
python-dotenv==1.0.1
pydantic==2.7.4
httpx==0.27.0
transformers==4.42.3
torch>=2.0.0
psutil==5.9.8
requests==2.31.0
wandb==0.17.0
```

Python version: **3.10 or higher**

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👤 Author

**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University, Dhaka  
📧 musa.islam.fahad@gmail.com  
🌐 [Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad) · [LinkedIn](https://linkedin.com/in/md-musa-islam-fahad-b18759249)

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Sentence Transformers](https://sbert.net/) for the `all-MiniLM-L6-v2` embedding model
- [ChromaDB](https://www.trychroma.com/) for the lightweight, persistent vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance REST framework
- [Streamlit](https://streamlit.io/) for the rapid dashboard framework
- [NLTK](https://www.nltk.org/) and [SpaCy](https://spacy.io/) for NLP preprocessing utilities
- [Plotly](https://plotly.com/python/) for interactive analytics charts
- [CodeAlpha](https://www.codealpha.tech/) for the internship opportunity and project brief

---

<div align="center">

Made with ❤️ as part of the **CodeAlpha AI/ML Internship**

**[⬆ Back to Top](#-visionchat-faq-assistant)**

</div>
