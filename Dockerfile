# ── VisionChat FAQ · Dockerfile ──────────────────────────────────────────────
# Build:  docker build -t visionchat-faq .
# Run:    docker run -p 8501:8501 visionchat-faq
# API:    docker run -p 8000:8000 visionchat-faq uvicorn api:app --host 0.0.0.0 --port 8000

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','stopwords','wordnet','averaged_perceptron_tagger']]"

# Download SpaCy model
RUN python -m spacy download en_core_web_sm || true

# Pre-download sentence-transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

COPY . .

RUN mkdir -p logs data/chroma_db

EXPOSE 8501 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
