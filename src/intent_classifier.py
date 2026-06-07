"""
intent_classifier.py — Lightweight intent detection layer.

Classifies user queries into high-level intents before semantic search,
enabling smarter routing and fallback handling.

Two modes:
  1. Embedding-based (default) — cosine similarity against intent exemplars
  2. Zero-shot (optional)      — facebook/bart-large-mnli via HuggingFace
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Intent definitions ────────────────────────────────────────────────────────

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "greeting": [
        "hello", "hi there", "hey", "good morning", "good evening",
        "howdy", "what's up", "greetings",
    ],
    "farewell": [
        "bye", "goodbye", "see you", "thanks bye", "take care",
        "farewell", "i'm done", "that's all",
    ],
    "gratitude": [
        "thank you", "thanks", "much appreciated", "thank you so much",
        "that helped", "great thanks",
    ],
    "complaint": [
        "this is broken", "not working", "terrible", "awful", "worst experience",
        "very disappointed", "unacceptable", "this is a problem",
    ],
    "pricing": [
        "how much does it cost", "what is the price", "pricing plans",
        "subscription cost", "how much is the plan", "fees",
    ],
    "account": [
        "my account", "login problem", "password reset", "sign up",
        "create account", "delete account", "profile settings",
    ],
    "billing": [
        "payment failed", "refund", "invoice", "charge", "cancel subscription",
        "billing issue", "double charged",
    ],
    "technical": [
        "not loading", "bug", "error", "crash", "slow", "not working",
        "broken feature", "api issue",
    ],
    "general_faq": [
        "how do i", "what is", "can i", "do you", "is there",
        "tell me about", "explain", "help me understand",
    ],
    "out_of_scope": [
        "what is the weather", "tell me a joke", "who is the president",
        "give me a recipe", "what movie should i watch",
    ],
}

INTENT_LABELS = list(INTENT_EXEMPLARS.keys())


@dataclass
class IntentResult:
    intent: str
    confidence: float
    is_faq_intent: bool
    requires_human: bool


# ── Classifier ────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Classifies user input into predefined intents using
    sentence-transformer embeddings over intent exemplars.
    """

    FAQ_INTENTS = {"pricing", "account", "billing", "technical", "general_faq"}
    HUMAN_HANDOFF_INTENTS = {"complaint"}

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : SentenceTransformer | None
            Pass the already-loaded model to avoid double loading.
            If None, classifier falls back to keyword matching.
        """
        self._model = model
        self._exemplar_embeddings: dict[str, np.ndarray] = {}
        self._ready = False

    def build(self) -> None:
        """Pre-compute exemplar embeddings."""
        if self._model is None:
            self._ready = False
            return

        for intent, exemplars in INTENT_EXEMPLARS.items():
            embs = self._model.encode(exemplars, show_progress_bar=False)
            self._exemplar_embeddings[intent] = embs.mean(axis=0, keepdims=True)

        self._ready = True

    def classify(self, text: str) -> IntentResult:
        """Classify text into an intent."""
        if self._ready and self._model is not None:
            return self._classify_embedding(text)
        return self._classify_keyword(text)

    def _classify_embedding(self, text: str) -> IntentResult:
        q_emb = self._model.encode([text], show_progress_bar=False)
        scores = {}
        for intent, mean_emb in self._exemplar_embeddings.items():
            sim = cosine_similarity(q_emb, mean_emb)[0][0]
            scores[intent] = float(sim)

        best_intent = max(scores, key=scores.__getitem__)
        confidence  = scores[best_intent]

        return IntentResult(
            intent=best_intent,
            confidence=round(confidence, 4),
            is_faq_intent=best_intent in self.FAQ_INTENTS,
            requires_human=best_intent in self.HUMAN_HANDOFF_INTENTS and confidence > 0.60,
        )

    def _classify_keyword(self, text: str) -> IntentResult:
        """Simple keyword fallback when embeddings are unavailable."""
        text_lower = text.lower()
        for intent, exemplars in INTENT_EXEMPLARS.items():
            for kw in exemplars:
                if kw in text_lower:
                    return IntentResult(
                        intent=intent,
                        confidence=0.70,
                        is_faq_intent=intent in self.FAQ_INTENTS,
                        requires_human=intent in self.HUMAN_HANDOFF_INTENTS,
                    )
        return IntentResult(
            intent="general_faq",
            confidence=0.50,
            is_faq_intent=True,
            requires_human=False,
        )
