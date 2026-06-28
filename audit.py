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
                text TEXT,
                attribution TEXT NOT NULL,
                confidence REAL NOT NULL,
                llm_score REAL NOT NULL,
                stylometric_score REAL NOT NULL,
                marker_score REAL,
                status TEXT NOT NULL,
                appeal_reasoning TEXT
            )
            """
        )
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(log_entries)")}
        if "marker_score" not in existing_columns:
            conn.execute("ALTER TABLE log_entries ADD COLUMN marker_score REAL")
        if "text" not in existing_columns:
            conn.execute("ALTER TABLE log_entries ADD COLUMN text TEXT")


def log_submission(
    content_id: str,
    creator_id: str,
    text: str,
    attribution: str,
    confidence: float,
    llm_score: float,
    stylometric_score: float,
    marker_score: float,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO log_entries (
                content_id, creator_id, timestamp, text, attribution, confidence,
                llm_score, stylometric_score, marker_score, status, appeal_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'classified', NULL)
            """,
            (
                content_id,
                creator_id,
                datetime.now(timezone.utc).isoformat(),
                text,
                attribution,
                confidence,
                llm_score,
                stylometric_score,
                marker_score,
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


def get_analytics() -> dict:
    """
    Aggregate metrics over all log entries, for the Analytics Dashboard stretch
    feature. Complements GET /log (individual decisions) with a system-wide view:
    detection pattern (counts/percentages by attribution), appeal rate, and
    average confidence as a sanity-check metric.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT attribution, confidence, status FROM log_entries").fetchall()

    total = len(rows)
    if total == 0:
        return {
            "total_submissions": 0,
            "detection_pattern": {
                label: {"count": 0, "percentage": 0.0}
                for label in ("likely_ai", "uncertain", "likely_human")
            },
            "appeal_rate": 0.0,
            "average_confidence": 0.0,
        }

    counts = {"likely_ai": 0, "uncertain": 0, "likely_human": 0}
    appealed = 0
    confidence_sum = 0.0
    for row in rows:
        counts[row["attribution"]] += 1
        if row["status"] == "under_review":
            appealed += 1
        confidence_sum += row["confidence"]

    detection_pattern = {
        label: {"count": count, "percentage": round(100 * count / total, 1)}
        for label, count in counts.items()
    }

    return {
        "total_submissions": total,
        "detection_pattern": detection_pattern,
        "appeal_rate": round(appealed / total, 3),
        "average_confidence": round(confidence_sum / total, 3),
    }
