"""
app.py — VisionChat FAQ · Streamlit application.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import uuid
import time
from pathlib import Path
from collections import defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "utils"))

load_dotenv(ROOT / ".env.example")

from chatbot import FAQChatbot
from analytics_db import AnalyticsDB

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionChat FAQ",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Chat bubbles */
.user-bubble {
    background: #185FA5;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 60px;
    font-size: 15px;
    line-height: 1.5;
}
.bot-bubble {
    background: #1E1E1E;
    color: #FAFAF8;
    border: 1px solid #333;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 60px 6px 0;
    font-size: 15px;
    line-height: 1.6;
}
.meta-chip {
    font-size: 11px;
    color: #888;
    margin-top: 4px;
    margin-left: 4px;
}
.confidence-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 4px;
}
.alt-question {
    font-size: 13px;
    color: #185FA5;
    cursor: pointer;
    text-decoration: underline;
    margin: 3px 0;
}
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 600; }
div[data-testid="metric-container"] {
    background: #1E1E1E;
    border: 1px solid #2A2A2A;
    border-radius: 10px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

FAQ_PATH = str(ROOT / "data" / "faqs.json")

# ── Session init ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_ids" not in st.session_state:
    st.session_state.query_ids = []
if "last_query_id" not in st.session_state:
    st.session_state.last_query_id = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Loading FAQ AI engine …")
def get_chatbot():
    bot = FAQChatbot(faq_path=FAQ_PATH, top_k=3)
    bot.load()
    return bot

@st.cache_resource
def get_db():
    return AnalyticsDB()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(bot: FAQChatbot):
    with st.sidebar:
        st.markdown("## 💬 VisionChat FAQ")
        st.caption(f"Session `{st.session_state.session_id}`")
        st.markdown("---")

        st.markdown("### 📂 Browse FAQs")
        categories = ["All"] + bot.categories
        selected_cat = st.selectbox("Filter by category", categories)

        faqs_to_show = bot.all_faqs
        if selected_cat != "All":
            faqs_to_show = [f for f in faqs_to_show if f["category"] == selected_cat]

        with st.expander(f"📋 {len(faqs_to_show)} FAQs", expanded=False):
            for faq in faqs_to_show:
                if st.button(f"❓ {faq['question']}", key=f"faq_{faq['id']}", use_container_width=True):
                    st.session_state.pending_input = faq["question"]
                    st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Example questions")
        examples = [
            "How do I reset my password?",
            "What is your refund policy?",
            "Do you have a mobile app?",
            "How do I cancel my subscription?",
            "Is my data GDPR compliant?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.pending_input = ex
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_ids = []
            st.session_state.last_query_id = None
            bot.clear_history()
            st.rerun()

        st.markdown("---")
        st.caption(f"📚 {bot._embedder.get_faq_count()} FAQs indexed")
        st.caption(f"🔧 Engine: sentence-transformers")
        st.caption("Built by **Md. Musa Islam Fahad**")
        st.caption("[GitHub](https://github.com/MusaIslamFahad) · [Portfolio](https://musaislamfahad.vercel.app)")


# ── Chat tab ──────────────────────────────────────────────────────────────────
def render_chat(bot: FAQChatbot, db: AnalyticsDB):
    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div class="bot-bubble">
        👋 <strong>Hello! I'm your FAQ Assistant.</strong><br>
        Ask me anything about accounts, billing, technical issues, shipping, privacy, or our features.<br><br>
        You can also browse FAQs by category in the sidebar, or click an example question to get started.
        </div>
        """, unsafe_allow_html=True)

    # Render message history
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            _render_bot_message(msg, i, db)

    # Feedback for last response
    if st.session_state.last_query_id and st.session_state.messages:
        last_bot = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
        if last_bot:
            c1, c2, _ = st.columns([1, 1, 6])
            with c1:
                if st.button("👍 Helpful", key="fb_yes"):
                    db.log_feedback(st.session_state.last_query_id, 1)
                    st.success("Thanks for your feedback!")
            with c2:
                if st.button("👎 Not helpful", key="fb_no"):
                    db.log_feedback(st.session_state.last_query_id, 0)
                    st.info("Thanks — we'll use this to improve.")

    # Input area
    st.markdown("---")
    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            "Ask a question …",
            value=st.session_state.pending_input,
            key="chat_input",
            placeholder="e.g. How do I cancel my subscription?",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("Send ➤", type="primary", use_container_width=True)

    if st.session_state.pending_input:
        st.session_state.pending_input = ""

    if (send or user_input) and user_input.strip():
        _process_message(user_input.strip(), bot, db)
        st.rerun()


def _render_bot_message(msg: dict, idx: int, db: AnalyticsDB):
    content = msg["content"]
    meta = msg.get("meta", {})
    conf = meta.get("confidence", 0)
    cat  = meta.get("category", "")
    mq   = meta.get("matched_question", "")
    mt   = meta.get("match_type", "")
    alts = meta.get("alternatives", [])
    rt   = meta.get("response_time_ms", 0)

    conf_color = "#27AE60" if conf >= 0.65 else "#F39C12" if conf >= 0.35 else "#E74C3C"
    conf_pct   = int(conf * 100)

    st.markdown(f'<div class="bot-bubble">🤖 {content}</div>', unsafe_allow_html=True)

    meta_parts = []
    if cat:     meta_parts.append(f"📂 {cat}")
    if conf:    meta_parts.append(f"Confidence: {conf_pct}%")
    if rt:      meta_parts.append(f"⚡ {rt:.0f}ms")
    if mt:      meta_parts.append(f"[{mt}]")

    if meta_parts:
        st.markdown(
            f'<div class="meta-chip">{" · ".join(meta_parts)}</div>',
            unsafe_allow_html=True,
        )
        # Confidence bar
        st.markdown(
            f'<div class="confidence-bar" style="width:{conf_pct}%;background:{conf_color};"></div>',
            unsafe_allow_html=True,
        )

    if mq:
        st.caption(f'📌 Matched: *"{mq}"*')

    if alts:
        st.markdown("**Related questions:**")
        for alt in alts:
            if st.button(f"→ {alt}", key=f"alt_{idx}_{alt[:30]}"):
                st.session_state.pending_input = alt
                st.rerun()


def _process_message(text: str, bot: FAQChatbot, db: AnalyticsDB):
    st.session_state.messages.append({"role": "user", "content": text})

    with st.spinner("Thinking …"):
        result = bot.chat(text)

    query_id = db.log_query(
        session_id=st.session_state.session_id,
        user_query=text,
        intent=result.intent,
        intent_confidence=result.confidence,
        match_type=result.match_type,
        match_score=result.confidence,
        matched_question=result.matched_question,
        category=result.category,
        response_time_ms=result.response_time_ms,
        requires_human=result.requires_human,
    )

    st.session_state.last_query_id = query_id
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "meta": {
            "confidence": result.confidence,
            "category": result.category,
            "matched_question": result.matched_question,
            "match_type": result.match_type,
            "alternatives": result.alternative_questions,
            "response_time_ms": result.response_time_ms,
        },
    })


# ── FAQ Browser tab ───────────────────────────────────────────────────────────
def render_faq_browser(bot: FAQChatbot):
    st.markdown("### 📖 Full FAQ Database")

    col_search, col_cat = st.columns([3, 1])
    with col_search:
        search_q = st.text_input("🔍 Search FAQs", placeholder="Type to filter …")
    with col_cat:
        cats = ["All"] + bot.categories
        cat_filter = st.selectbox("Category", cats, key="browser_cat")

    faqs = bot.all_faqs
    if cat_filter != "All":
        faqs = [f for f in faqs if f["category"] == cat_filter]
    if search_q:
        faqs = [f for f in faqs if search_q.lower() in f["question"].lower()
                                 or search_q.lower() in f["answer"].lower()]

    st.caption(f"Showing {len(faqs)} FAQs")
    for faq in faqs:
        with st.expander(f"[{faq['category']}] {faq['question']}"):
            st.markdown(faq["answer"])
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(f"ID: `{faq['id']}`")
            with col2:
                if st.button("Ask this →", key=f"ask_{faq['id']}"):
                    st.session_state.pending_input = faq["question"]
                    st.rerun()


# ── Analytics tab ─────────────────────────────────────────────────────────────
def render_analytics(db: AnalyticsDB):
    st.markdown("### 📊 Analytics Dashboard")
    summary = db.get_summary()

    if summary["total_queries"] == 0:
        st.info("No queries logged yet. Start chatting to see analytics here!")
        return

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries", summary["total_queries"])
    c2.metric("Avg Confidence", f"{summary['avg_confidence']:.0%}")
    c3.metric("Avg Response", f"{summary['avg_response_ms']:.0f}ms")
    c4.metric("No-Match", summary["no_match_count"])
    c5.metric("Human Handoffs", summary["human_handoff_count"])

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        # Intent distribution
        intent_data = summary.get("intent_distribution", {})
        if intent_data:
            df_intent = pd.DataFrame(
                intent_data.items(), columns=["Intent", "Count"]
            ).sort_values("Count", ascending=True)
            fig = px.bar(df_intent, x="Count", y="Intent", orientation="h",
                         title="Query Intent Distribution", color="Count",
                         color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#FAFAF8", showlegend=False,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Category distribution
        cat_data = summary.get("category_distribution", {})
        if cat_data:
            df_cat = pd.DataFrame(cat_data.items(), columns=["Category", "Count"])
            fig2 = px.pie(df_cat, names="Category", values="Count",
                          title="Queries by FAQ Category",
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#FAFAF8",
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    # Recent queries table
    st.markdown("### 🕐 Recent Queries")
    df = db.get_queries_df(limit=50)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
        df["match_score"] = df["match_score"].apply(lambda x: f"{x:.0%}" if x else "—")
        show_cols = ["timestamp", "user_query", "intent", "category",
                     "match_score", "match_type", "response_time_ms"]
        st.dataframe(df[show_cols].head(30), use_container_width=True)

    # Unanswered questions
    st.markdown("### ❓ Unanswered Questions")
    unanswered = db.get_unanswered(limit=15)
    if not unanswered.empty:
        st.dataframe(unanswered, use_container_width=True)
        st.caption("These queries had no confident match — add them to your FAQ dataset to improve coverage.")
    else:
        st.success("All queries were answered! 🎉")


# ── About tab ─────────────────────────────────────────────────────────────────
def render_about():
    st.markdown("""
## 💬 VisionChat FAQ AI

A production-grade **FAQ Chatbot** built for the **CodeAlpha AI Internship (Task 2)**.

### Tech Stack

| Component | Tool |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector database | ChromaDB (persistent, local) |
| NLP preprocessing | NLTK + SpaCy (tokenisation, lemmatisation, NER) |
| Intent classifier | Embedding-based cosine similarity over intent exemplars |
| Analytics store | SQLite (zero-cost, zero-infra) |
| Backend API | FastAPI REST (production-grade) |
| Frontend UI | Streamlit |
| Deployment | Hugging Face Spaces / Streamlit Cloud |

### Architecture
```
User Query
    │
    ▼
NLP Preprocessor  ← clean, tokenise, lemmatise (NLTK + SpaCy)
    │
    ▼
Intent Classifier ← embedding-based intent routing (8 intents)
    │
    ├── greeting / farewell / gratitude → template response
    ├── complaint                       → human handoff message
    ├── out_of_scope                    → graceful deflection
    │
    └── FAQ intent
            │
            ▼
    Sentence Transformer  ← all-MiniLM-L6-v2 encodes query
            │
            ▼
    ChromaDB vector search  ← cosine similarity over FAQ embeddings
            │
            ▼
    Confidence gating  ← high (≥65%) / low / no-match routing
            │
            ▼
    Structured Response  ← answer + alternatives + metadata
            │
            ▼
    SQLite Analytics Logger  ← every query recorded
            │
            ▼
    Streamlit UI  ← chat bubbles, confidence bar, alternatives
```

### Features
- 🧠 Semantic understanding via Sentence Transformers (not keyword matching)
- 🗂️ ChromaDB vector store with persistent indexing
- 🎯 Intent classification with 8 intent categories
- 📊 Live analytics dashboard (intent dist, category dist, unanswered)
- 👍 Per-response feedback (helpful / not helpful)
- 🔌 FastAPI REST backend with full Swagger docs
- 📖 Browseable FAQ database with search and category filter
- 💾 SQLite query logging — zero infrastructure, zero cost
- 🌐 Deployable for free on HF Spaces or Streamlit Cloud

### Author
**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University  
[Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad) · [LinkedIn](https://linkedin.com/in/md-musa-islam-fahad-b18759249)  
CodeAlpha AI Internship · 2025
""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    bot = get_chatbot()
    db  = get_db()

    render_sidebar(bot)

    st.title("💬 VisionChat FAQ Assistant")
    st.caption("Semantic FAQ search · Sentence Transformers + ChromaDB · CodeAlpha AI Internship Task 2")

    tab1, tab2, tab3, tab4 = st.tabs(["🗨️ Chat", "📖 FAQ Browser", "📊 Analytics", "ℹ️ About"])

    with tab1:
        render_chat(bot, db)
    with tab2:
        render_faq_browser(bot)
    with tab3:
        render_analytics(db)
    with tab4:
        render_about()


if __name__ == "__main__":
    main()
