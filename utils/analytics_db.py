"""
analytics_db.py — SQLite-backed analytics logger.

Records every query, intent, confidence score, and response time.
Powers the analytics dashboard in the Streamlit app.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = str(Path(__file__).parent.parent / "logs" / "analytics.db")


class AnalyticsDB:
    """Lightweight SQLite analytics store."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT,
                    timestamp       REAL,
                    user_query      TEXT,
                    intent          TEXT,
                    intent_confidence REAL,
                    match_type      TEXT,
                    match_score     REAL,
                    matched_question TEXT,
                    category        TEXT,
                    response_time_ms REAL,
                    requires_human  INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id    INTEGER,
                    rating      INTEGER,   -- 1=helpful, 0=not helpful
                    comment     TEXT,
                    timestamp   REAL,
                    FOREIGN KEY (query_id) REFERENCES queries(id)
                )
            """)
            conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_query(
        self,
        session_id: str,
        user_query: str,
        intent: str,
        intent_confidence: float,
        match_type: str,
        match_score: float,
        matched_question: Optional[str],
        category: Optional[str],
        response_time_ms: float,
        requires_human: bool,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO queries
                (session_id, timestamp, user_query, intent, intent_confidence,
                 match_type, match_score, matched_question, category,
                 response_time_ms, requires_human)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                session_id, time.time(), user_query, intent, intent_confidence,
                match_type, match_score, matched_question, category,
                response_time_ms, int(requires_human),
            ))
            conn.commit()
            return cur.lastrowid

    def log_feedback(self, query_id: int, rating: int, comment: str = "") -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO feedback (query_id, rating, comment, timestamp)
                VALUES (?,?,?,?)
            """, (query_id, rating, comment, time.time()))
            conn.commit()

    # ── Read / analytics ──────────────────────────────────────────────────────

    def get_queries_df(self, limit: int = 500) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql(
                f"SELECT * FROM queries ORDER BY timestamp DESC LIMIT {limit}",
                conn,
            )

    def get_summary(self) -> dict:
        with self._conn() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM queries")
            total = cur.fetchone()[0]

            cur = conn.execute("SELECT AVG(match_score) FROM queries WHERE match_type != 'no_match'")
            avg_score = cur.fetchone()[0] or 0.0

            cur = conn.execute("SELECT AVG(response_time_ms) FROM queries")
            avg_rt = cur.fetchone()[0] or 0.0

            cur = conn.execute("SELECT COUNT(*) FROM queries WHERE match_type = 'no_match'")
            no_match = cur.fetchone()[0]

            cur = conn.execute("SELECT COUNT(*) FROM queries WHERE requires_human = 1")
            human = cur.fetchone()[0]

            cur = conn.execute("""
                SELECT intent, COUNT(*) as cnt
                FROM queries GROUP BY intent ORDER BY cnt DESC
            """)
            intent_dist = dict(cur.fetchall())

            cur = conn.execute("""
                SELECT category, COUNT(*) as cnt
                FROM queries WHERE category IS NOT NULL
                GROUP BY category ORDER BY cnt DESC
            """)
            cat_dist = dict(cur.fetchall())

        return {
            "total_queries": total,
            "avg_confidence": round(avg_score, 3),
            "avg_response_ms": round(avg_rt, 1),
            "no_match_count": no_match,
            "human_handoff_count": human,
            "intent_distribution": intent_dist,
            "category_distribution": cat_dist,
        }

    def get_unanswered(self, limit: int = 20) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql("""
                SELECT user_query, timestamp FROM queries
                WHERE match_type = 'no_match'
                ORDER BY timestamp DESC
                LIMIT ?
            """, conn, params=(limit,))

    def get_feedback_df(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql("""
                SELECT f.rating, f.comment, f.timestamp, q.user_query, q.category
                FROM feedback f
                JOIN queries q ON f.query_id = q.id
                ORDER BY f.timestamp DESC
            """, conn)
