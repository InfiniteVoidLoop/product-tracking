"""
src/database.py
===============
SQLite-backed database manager for the Conveyor Belt CV System.

Schema
------
products
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    track_id    INTEGER NOT NULL
    timestamp   REAL    NOT NULL          -- Unix epoch (seconds)
    status      TEXT    NOT NULL          -- "Normal" | "Defective" | "Pending"
    confidence  REAL                      -- Classifier confidence
    frame_idx   INTEGER
    counted_at  TEXT                      -- ISO-8601 datetime string

alerts
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    product_id  INTEGER REFERENCES products(id)
    alert_type  TEXT    NOT NULL          -- "defect_detected" | "throughput_spike" etc.
    message     TEXT
    timestamp   REAL    NOT NULL

throughput_log
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    minute      TEXT    NOT NULL          -- "YYYY-MM-DD HH:MM"
    count       INTEGER NOT NULL DEFAULT 0
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProductRecord:
    track_id: int
    status: str = "Pending"
    confidence: float = 0.0
    frame_idx: int = 0
    timestamp: float = 0.0
    id: Optional[int] = None
    counted_at: Optional[str] = None


@dataclass
class AlertRecord:
    product_id: Optional[int]
    alert_type: str
    message: str
    timestamp: float = 0.0
    id: Optional[int] = None


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class Database:
    """
    Thread-safe SQLite database manager.

    Args:
        db_path: Path to the SQLite database file (created if absent).
    """

    def __init__(self, db_path: str | Path = "data/conveyor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id   INTEGER NOT NULL,
                    timestamp  REAL    NOT NULL,
                    status     TEXT    NOT NULL DEFAULT 'Pending',
                    confidence REAL    DEFAULT 0.0,
                    frame_idx  INTEGER DEFAULT 0,
                    counted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id INTEGER REFERENCES products(id),
                    alert_type TEXT    NOT NULL,
                    message    TEXT    DEFAULT '',
                    timestamp  REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS throughput_log (
                    id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    minute TEXT    NOT NULL UNIQUE,
                    count  INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_products_timestamp
                    ON products(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                    ON alerts(timestamp);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Return a thread-safe connection context manager."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def insert_product(self, record: ProductRecord) -> int:
        """Insert a product record and return its auto-incremented id."""
        ts = record.timestamp or time.time()
        counted_at = record.counted_at or datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO products (track_id, timestamp, status, confidence, frame_idx, counted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.track_id,
                    ts,
                    record.status,
                    record.confidence,
                    record.frame_idx,
                    counted_at,
                ),
            )
            row_id = cur.lastrowid

        # Update throughput log
        minute_key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        self._upsert_throughput(minute_key)

        return row_id

    def update_product_status(
        self,
        product_id: int,
        status: str,
        confidence: float = 0.0,
    ):
        """Update the status (and classifier confidence) for an existing product record."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE products SET status=?, confidence=? WHERE id=?",
                (status, confidence, product_id),
            )

    def insert_alert(self, record: AlertRecord) -> int:
        ts = record.timestamp or time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts (product_id, alert_type, message, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (record.product_id, record.alert_type, record.message, ts),
            )
            return cur.lastrowid

    def _upsert_throughput(self, minute_key: str):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO throughput_log (minute, count)
                VALUES (?, 1)
                ON CONFLICT(minute) DO UPDATE SET count = count + 1
                """,
                (minute_key,),
            )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_total_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM products").fetchone()
        return row[0] if row else 0

    def get_status_counts(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM products GROUP BY status"
            ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def get_recent_products(self, n: int = 50) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM products ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_alerts(self, n: int = 50) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*, p.track_id
                FROM alerts a
                LEFT JOIN products p ON a.product_id = p.id
                ORDER BY a.timestamp DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_throughput_by_minute(self, last_n_minutes: int = 60) -> List[dict]:
        """Return throughput counts for the last N minutes."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT minute, count
                FROM throughput_log
                ORDER BY minute DESC
                LIMIT ?
                """,
                (last_n_minutes,),
            ).fetchall()
        result = [dict(r) for r in rows]
        result.reverse()  # chronological order
        return result

    def get_stats_summary(self) -> Dict:
        """Convenience method returning a stats dict for the dashboard."""
        status_counts = self.get_status_counts()
        total = self.get_total_count()
        return {
            "total": total,
            "normal": status_counts.get("Normal", 0),
            "defective": status_counts.get("Defective", 0),
            "pending": status_counts.get("Pending", 0),
        }

    def clear_all(self):
        """Wipe all data (for testing)."""
        with self._connect() as conn:
            conn.executescript(
                "DELETE FROM alerts; DELETE FROM products; DELETE FROM throughput_log;"
            )

    def close(self):
        pass  # Connections are per-operation; nothing to close.
