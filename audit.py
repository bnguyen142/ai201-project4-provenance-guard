import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_FILE = "logs/audit.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS log_entries (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                attribution TEXT NOT NULL,
                confidence REAL NOT NULL,
                llm_score REAL NOT NULL,
                stylometric_score REAL NOT NULL,
                status TEXT NOT NULL,
                appeal_reasoning TEXT
            )
            """
        )


def log_submission(
    content_id: str,
    creator_id: str,
    attribution: str,
    confidence: float,
    llm_score: float,
    stylometric_score: float,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO log_entries (
                content_id, creator_id, timestamp, attribution, confidence,
                llm_score, stylometric_score, status, appeal_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'classified', NULL)
            """,
            (
                content_id,
                creator_id,
                datetime.now(timezone.utc).isoformat(),
                attribution,
                confidence,
                llm_score,
                stylometric_score,
            ),
        )


def record_appeal(content_id: str, creator_reasoning: str) -> bool:
    """
    Update the existing log entry for `content_id` in place — never insert a
    new row (CLAUDE.md guardrail #6). Returns False if content_id isn't found
    or has already been appealed.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT status FROM log_entries WHERE content_id = ?", (content_id,)
        ).fetchone()
        if row is None or row["status"] == "under_review":
            return False

        conn.execute(
            """
            UPDATE log_entries
            SET status = 'under_review', appeal_reasoning = ?
            WHERE content_id = ?
            """,
            (creator_reasoning, content_id),
        )
        return True


def get_log() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM log_entries ORDER BY timestamp ASC"
        ).fetchall()
        return [dict(row) for row in rows]
