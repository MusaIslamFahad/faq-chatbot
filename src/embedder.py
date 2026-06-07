"""
embedder.py — Semantic embedding engine.

Uses sentence-transformers (all-MiniLM-L6-v2) to encode FAQ questions
into dense vectors, stored in ChromaDB for fast cosine similarity search.
Falls back to TF-IDF if the transformer model cannot be loaded.
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

# ── Sentence Transformers ─────────────────────────────────────────────────────
_st_available = False
try:
    from sentence_transformers import SentenceTransformer
    _st_available = True
except ImportError:
    pass

# ── ChromaDB ─────────────────────────────────────────────────────────────────
_chroma_available = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _chroma_available = True
except ImportError:
    pass

# ── TF-IDF fallback ───────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessor import cached_preprocess

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "all-MiniLM-L6-v2"
COLLECTION_NAME = "faq_embeddings"
CHROMA_DIR    = str(Path(__file__).parent.parent / "data" / "chroma_db")


# ── Main embedding class ──────────────────────────────────────────────────────

class FAQEmbedder:
    """
    Semantic FAQ search engine.

    Priority:
      1. Sentence Transformers + ChromaDB (best accuracy, industry-standard)
      2. TF-IDF + cosine similarity (fallback, zero extra deps)

    Parameters
    ----------
    faq_path : str
        Path to the faqs.json data file.
    top_k : int
        Number of candidates to return from similarity search.
    similarity_threshold : float
        Minimum cosine similarity to be considered a valid match.
    persist_dir : str
        Directory for ChromaDB persistence.
    """

    def __init__(
        self,
        faq_path: str,
        top_k: int = 5,
        similarity_threshold: float = 0.30,
        persist_dir: str = CHROMA_DIR,
    ):
        self.faq_path = faq_path
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.persist_dir = persist_dir

        self._faqs: list[dict] = []
        self._model: Optional[SentenceTransformer] = None
        self._collection = None
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._use_transformers = False
        self._loaded = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> str:
        """Load FAQs, model, and build/restore vector index. Returns mode string."""
        self._faqs = self._load_faqs()

        if _st_available:
            try:
                self._model = SentenceTransformer(MODEL_NAME)
                self._use_transformers = True
                self._setup_chroma()
                self._loaded = True
                return "sentence-transformers + ChromaDB"
            except Exception as e:
                print(f"[Embedder] Transformer load failed ({e}), falling back to TF-IDF.")

        # Fallback
        self._setup_tfidf()
        self._loaded = True
        return "TF-IDF (fallback)"

    def _load_faqs(self) -> list[dict]:
        with open(self.faq_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _faq_corpus_hash(self) -> str:
        content = json.dumps(self._faqs, sort_keys=True).encode()
        return hashlib.md5(content).hexdigest()[:12]

    # ── ChromaDB setup ────────────────────────────────────────────────────────

    def _setup_chroma(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)

        if _chroma_available:
            client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            # In-memory fallback (no persistence)
            client = chromadb.Client()

        # Check if collection already indexed with same FAQ data
        corpus_hash = self._faq_corpus_hash()
        collection_name = f"{COLLECTION_NAME}_{corpus_hash}"

        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            self._collection = client.get_collection(collection_name)
            return

        # Build new collection
        self._collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        questions = [f["question"] for f in self._faqs]
        embeddings = self._model.encode(questions, show_progress_bar=False).tolist()

        self._collection.add(
            ids=[f["id"] for f in self._faqs],
            embeddings=embeddings,
            documents=questions,
            metadatas=[
                {"category": f["category"], "answer": f["answer"], "question": f["question"]}
                for f in self._faqs
            ],
        )

    # ── TF-IDF setup ──────────────────────────────────────────────────────────

    def _setup_tfidf(self) -> None:
        corpus = [cached_preprocess(f["question"]) for f in self._faqs]
        self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Return top-k FAQ matches for a query.
        Each result: {id, question, answer, category, score, rank}
        """
        if not self._loaded:
            self.load()

        k = top_k or self.top_k

        if self._use_transformers and self._collection is not None:
            return self._search_chroma(query, k)
        else:
            return self._search_tfidf(query, k)

    def _search_chroma(self, query: str, k: int) -> list[dict]:
        q_emb = self._model.encode([query], show_progress_bar=False).tolist()
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(k, len(self._faqs)),
            include=["metadatas", "distances", "documents"],
        )

        matches = []
        for i, (meta, dist) in enumerate(
            zip(results["metadatas"][0], results["distances"][0])
        ):
            # ChromaDB cosine distance: score = 1 - distance
            score = float(1.0 - dist)
            if score < self.similarity_threshold:
                continue
            matches.append({
                "rank": i + 1,
                "question": meta["question"],
                "answer": meta["answer"],
                "category": meta["category"],
                "score": round(score, 4),
                "match_type": "semantic",
            })
        return matches

    def _search_tfidf(self, query: str, k: int) -> list[dict]:
        q_vec = self._tfidf_vectorizer.transform([cached_preprocess(query)])
        sims  = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]

        matches = []
        for rank, idx in enumerate(top_idx):
            score = float(sims[idx])
            if score < self.similarity_threshold:
                continue
            faq = self._faqs[idx]
            matches.append({
                "rank": rank + 1,
                "question": faq["question"],
                "answer": faq["answer"],
                "category": faq["category"],
                "score": round(score, 4),
                "match_type": "tfidf",
            })
        return matches

    # ── FAQ access ────────────────────────────────────────────────────────────

    def get_all_faqs(self) -> list[dict]:
        return self._faqs

    def get_categories(self) -> list[str]:
        return sorted({f["category"] for f in self._faqs})

    def get_faq_count(self) -> int:
        return len(self._faqs)

    def add_faq(self, faq: dict) -> None:
        """Dynamically add a new FAQ and update the index."""
        self._faqs.append(faq)
        if self._use_transformers and self._collection:
            emb = self._model.encode([faq["question"]], show_progress_bar=False).tolist()
            self._collection.add(
                ids=[faq["id"]],
                embeddings=emb,
                documents=[faq["question"]],
                metadatas=[{"category": faq["category"],
                            "answer": faq["answer"],
                            "question": faq["question"]}],
            )
        elif self._tfidf_vectorizer:
            self._setup_tfidf()  # rebuild TF-IDF matrix
