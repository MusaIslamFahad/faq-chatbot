"""
chatbot.py — Core FAQ chatbot engine.

Orchestrates:
  preprocessor → intent classifier → semantic search → response generator
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from preprocessor import preprocess, extract_entities_spacy
from embedder import FAQEmbedder
from intent_classifier import IntentClassifier, IntentResult

# ── Response templates ────────────────────────────────────────────────────────

GREETING_RESPONSES = [
    "Hello! 👋 I'm your FAQ assistant. Ask me anything about our products and services!",
    "Hi there! 😊 Happy to help. What would you like to know?",
    "Hey! Great to see you. I can answer questions about accounts, billing, technical issues, and more.",
]

FAREWELL_RESPONSES = [
    "Goodbye! Feel free to come back anytime. 👋",
    "See you later! Hope I was helpful. 😊",
    "Take care! Don't hesitate to ask if you have more questions.",
]

GRATITUDE_RESPONSES = [
    "You're welcome! Is there anything else I can help you with? 😊",
    "Happy to help! Feel free to ask more questions.",
    "Glad I could assist! Anything else on your mind?",
]

NO_MATCH_RESPONSES = [
    "I couldn't find a confident match for your question in our FAQ database. Could you rephrase it, or try asking about a specific topic like billing, accounts, or technical support?",
    "Hmm, I'm not sure about that one. Try asking something like 'How do I reset my password?' or 'What is your refund policy?'",
    "That question is outside what I've been trained on. You can reach our support team at support@example.com for personalised help.",
]

COMPLAINT_RESPONSE = (
    "I'm really sorry to hear you're having a frustrating experience. 😟 "
    "I've noted your concern. For urgent issues, please contact our support team directly at "
    "support@example.com or via live chat — they'll prioritise your case."
)

LOW_CONFIDENCE_PREFIX = (
    "I found a possible match, but I'm not very confident it answers your question. "
    "You may want to rephrase or contact support if this doesn't help:\n\n"
)

import random
random.seed(42)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str        # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[str] = None
    confidence: Optional[float] = None
    matched_question: Optional[str] = None
    category: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class ChatResponse:
    answer: str
    intent: str
    confidence: float
    matched_question: Optional[str]
    category: Optional[str]
    alternative_questions: list[str]
    response_time_ms: float
    match_type: str       # "exact", "semantic", "tfidf", "intent", "no_match"
    requires_human: bool


# ── Chatbot ───────────────────────────────────────────────────────────────────

class FAQChatbot:
    """
    Production-grade FAQ chatbot.

    Pipeline:
      1. Preprocess input text (clean, normalise)
      2. Classify intent
      3. Route: greeting/farewell/gratitude → template response
               complaint → human handoff
               faq intent → semantic search → ranked answer
      4. Confidence-gated response with fallback messaging
      5. Return structured ChatResponse
    """

    HIGH_CONFIDENCE_THRESHOLD  = 0.65
    LOW_CONFIDENCE_THRESHOLD   = 0.35

    def __init__(self, faq_path: str, top_k: int = 3):
        self.faq_path = faq_path
        self.top_k    = top_k

        self._embedder   = FAQEmbedder(faq_path=faq_path, top_k=top_k + 2)
        self._classifier: Optional[IntentClassifier] = None
        self._history: list[ChatMessage] = []
        self._mode: str = "not loaded"
        self._loaded = False

    # ── Initialisation ────────────────────────────────────────────────────────

    def load(self) -> str:
        """Load all components. Returns mode description string."""
        self._mode = self._embedder.load()

        # Share the transformer model with intent classifier to avoid double load
        model = self._embedder._model
        self._classifier = IntentClassifier(model=model)
        self._classifier.build()

        self._loaded = True
        return self._mode

    # ── Main chat method ──────────────────────────────────────────────────────

    def chat(self, user_input: str) -> ChatResponse:
        """Process user input and return a structured response."""
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        user_input = user_input.strip()

        if not user_input:
            return self._make_response(
                "Please type a question and I'll do my best to help!",
                intent="empty", confidence=1.0, match_type="intent",
                elapsed=0.0,
            )

        # 1. Intent classification
        intent_result: IntentResult = self._classifier.classify(user_input)

        # 2. Route by intent
        response = self._route(user_input, intent_result)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        response.response_time_ms = round(elapsed_ms, 2)

        # 3. Record history
        self._history.append(ChatMessage(
            role="user",
            content=user_input,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
        ))
        self._history.append(ChatMessage(
            role="assistant",
            content=response.answer,
            intent=response.intent,
            confidence=response.confidence,
            matched_question=response.matched_question,
            category=response.category,
            response_time_ms=elapsed_ms,
        ))

        return response

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, text: str, intent: IntentResult) -> ChatResponse:

        # Greeting
        if intent.intent == "greeting":
            return self._make_response(
                random.choice(GREETING_RESPONSES),
                intent="greeting", confidence=intent.confidence,
                match_type="intent",
            )

        # Farewell
        if intent.intent == "farewell":
            return self._make_response(
                random.choice(FAREWELL_RESPONSES),
                intent="farewell", confidence=intent.confidence,
                match_type="intent",
            )

        # Gratitude
        if intent.intent == "gratitude":
            return self._make_response(
                random.choice(GRATITUDE_RESPONSES),
                intent="gratitude", confidence=intent.confidence,
                match_type="intent",
            )

        # Complaint → human handoff
        if intent.requires_human:
            return self._make_response(
                COMPLAINT_RESPONSE,
                intent="complaint", confidence=intent.confidence,
                match_type="intent", requires_human=True,
            )

        # Out of scope
        if intent.intent == "out_of_scope" and intent.confidence > 0.70:
            return self._make_response(
                "That question seems outside my knowledge area. I'm specialised in product FAQs. "
                "Try asking about billing, your account, technical issues, or our features!",
                intent="out_of_scope", confidence=intent.confidence,
                match_type="intent",
            )

        # FAQ search
        return self._faq_search(text, intent)

    def _faq_search(self, text: str, intent: IntentResult) -> ChatResponse:
        matches = self._embedder.search(text, top_k=self.top_k + 2)

        if not matches:
            return self._make_response(
                random.choice(NO_MATCH_RESPONSES),
                intent=intent.intent, confidence=0.0,
                match_type="no_match",
            )

        best = matches[0]
        score = best["score"]
        alternatives = [m["question"] for m in matches[1 : self.top_k]]

        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            answer = best["answer"]
            match_type = best.get("match_type", "semantic")
        elif score >= self.LOW_CONFIDENCE_THRESHOLD:
            answer = LOW_CONFIDENCE_PREFIX + best["answer"]
            match_type = "low_confidence"
        else:
            return self._make_response(
                random.choice(NO_MATCH_RESPONSES),
                intent=intent.intent, confidence=score,
                match_type="no_match",
            )

        return self._make_response(
            answer,
            intent=intent.intent,
            confidence=score,
            match_type=match_type,
            matched_question=best["question"],
            category=best["category"],
            alternatives=alternatives,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_response(
        answer: str,
        intent: str,
        confidence: float,
        match_type: str,
        elapsed: float = 0.0,
        matched_question: Optional[str] = None,
        category: Optional[str] = None,
        alternatives: Optional[list[str]] = None,
        requires_human: bool = False,
    ) -> ChatResponse:
        return ChatResponse(
            answer=answer,
            intent=intent,
            confidence=confidence,
            matched_question=matched_question,
            category=category,
            alternative_questions=alternatives or [],
            response_time_ms=elapsed,
            match_type=match_type,
            requires_human=requires_human,
        )

    # ── History & analytics ───────────────────────────────────────────────────

    def get_history(self) -> list[ChatMessage]:
        return self._history.copy()

    def clear_history(self) -> None:
        self._history.clear()

    def get_stats(self) -> dict:
        user_msgs = [m for m in self._history if m.role == "user"]
        asst_msgs = [m for m in self._history if m.role == "assistant"]
        confidences = [m.confidence for m in asst_msgs if m.confidence is not None]
        rtimes = [m.response_time_ms for m in asst_msgs if m.response_time_ms is not None]

        intent_dist: dict[str, int] = {}
        for m in user_msgs:
            if m.intent:
                intent_dist[m.intent] = intent_dist.get(m.intent, 0) + 1

        return {
            "total_messages": len(user_msgs),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
            "avg_response_ms": round(sum(rtimes) / len(rtimes), 1) if rtimes else 0,
            "intent_distribution": intent_dist,
            "faq_count": self._embedder.get_faq_count(),
            "mode": self._mode,
        }

    @property
    def categories(self) -> list[str]:
        return self._embedder.get_categories()

    @property
    def all_faqs(self) -> list[dict]:
        return self._embedder.get_all_faqs()
