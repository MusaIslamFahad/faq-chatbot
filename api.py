"""
api.py — FastAPI REST backend for VisionChat FAQ.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /chat              ← main chat endpoint
    GET  /faqs              ← list all FAQs
    GET  /faqs/{category}   ← FAQs by category
    GET  /categories        ← list categories
    POST /feedback          ← submit rating
    GET  /analytics/summary ← session analytics
    GET  /health            ← health check
"""

from __future__ import annotations

import sys
import uuid
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "utils"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chatbot import FAQChatbot
from analytics_db import AnalyticsDB

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VisionChat FAQ API",
    description="Production-grade FAQ chatbot REST API · CodeAlpha AI Internship Task 2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAQ_PATH = str(ROOT / "data" / "faqs.json")
_chatbot = FAQChatbot(faq_path=FAQ_PATH, top_k=3)
_db      = AnalyticsDB()

@app.on_event("startup")
async def startup():
    mode = _chatbot.load()
    print(f"[API] Chatbot loaded in mode: {mode}")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    matched_question: Optional[str]
    category: Optional[str]
    alternative_questions: list[str]
    response_time_ms: float
    match_type: str
    requires_human: bool
    session_id: str
    query_id: Optional[int]

class FeedbackRequest(BaseModel):
    query_id: int
    rating: int = Field(..., ge=0, le=1)
    comment: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        result = _chatbot.chat(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    query_id = _db.log_query(
        session_id=session_id,
        user_query=req.query,
        intent=result.intent,
        intent_confidence=result.confidence,
        match_type=result.match_type,
        match_score=result.confidence,
        matched_question=result.matched_question,
        category=result.category,
        response_time_ms=result.response_time_ms,
        requires_human=result.requires_human,
    )

    return ChatResponse(
        answer=result.answer,
        intent=result.intent,
        confidence=result.confidence,
        matched_question=result.matched_question,
        category=result.category,
        alternative_questions=result.alternative_questions,
        response_time_ms=result.response_time_ms,
        match_type=result.match_type,
        requires_human=result.requires_human,
        session_id=session_id,
        query_id=query_id,
    )


@app.get("/faqs")
def list_faqs():
    return {"faqs": _chatbot.all_faqs, "count": len(_chatbot.all_faqs)}


@app.get("/faqs/{category}")
def faqs_by_category(category: str):
    faqs = [f for f in _chatbot.all_faqs if f["category"].lower() == category.lower()]
    if not faqs:
        raise HTTPException(status_code=404, detail=f"No FAQs found for category: {category}")
    return {"category": category, "faqs": faqs, "count": len(faqs)}


@app.get("/categories")
def list_categories():
    return {"categories": _chatbot.categories}


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    _db.log_feedback(req.query_id, req.rating, req.comment)
    return {"status": "ok"}


@app.get("/analytics/summary")
def analytics_summary():
    return _db.get_summary()


@app.get("/analytics/queries")
def analytics_queries(limit: int = 100):
    df = _db.get_queries_df(limit=limit)
    return df.to_dict(orient="records")


@app.get("/analytics/unanswered")
def analytics_unanswered():
    df = _db.get_unanswered()
    return df.to_dict(orient="records")
