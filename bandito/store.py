"""SQLite WAL durability layer — crash-safe event storage.

Events are written here immediately after pull(). Background thread
flushes to cloud. If the process crashes, pending events survive and
are retried on next connect().
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    local_event_uuid TEXT PRIMARY KEY,
    bandit_id        INTEGER NOT NULL,
    arm_id           INTEGER NOT NULL,
    payload          TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    created_at       REAL NOT NULL,
    human_reward     REAL,
    graded_at        REAL
);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
"""

_MIGRATION_GRADING = [
    "ALTER TABLE events ADD COLUMN human_reward REAL",
    "ALTER TABLE events ADD COLUMN graded_at REAL",
]


class EventStore:
    """SQLite-backed event queue with WAL for crash resilience.

    Args:
        db_path: Path to SQLite file, or ":memory:" for testing.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def push(self, event: dict[str, Any]) -> None:
        """Insert a pending event."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO events "
                "(local_event_uuid, bandit_id, arm_id, payload, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (
                    event["local_event_uuid"],
                    event["bandit_id"],
                    event["arm_id"],
                    json.dumps(event),
                    time.time(),
                ),
            )
            self._conn.commit()

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return up to `limit` pending events (oldest first)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT payload FROM events WHERE status = 'pending' "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            )
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def mark_flushed(self, uuids: list[str]) -> None:
        """Mark events as successfully flushed to cloud."""
        if not uuids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in uuids)
            self._conn.execute(
                f"UPDATE events SET status = 'flushed' "
                f"WHERE local_event_uuid IN ({placeholders})",
                uuids,
            )
            self._conn.commit()

    def _migrate(self) -> None:
        """Apply schema migrations for existing databases."""
        for stmt in _MIGRATION_GRADING:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def ungraded(self, bandit_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return flushed events that haven't been graded yet."""
        with self._lock:
            if bandit_id is not None:
                cursor = self._conn.execute(
                    "SELECT payload FROM events "
                    "WHERE human_reward IS NULL AND status = 'flushed' "
                    "AND bandit_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (bandit_id, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT payload FROM events "
                    "WHERE human_reward IS NULL AND status = 'flushed' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def get_text(self, uuids: list[str]) -> dict[str, dict[str, str | None]]:
        """Look up query_text and response for a batch of UUIDs.

        Returns:
            Mapping of uuid → {"query_text": ..., "response": ...}
            Only includes UUIDs found in the store.
        """
        if not uuids:
            return {}
        with self._lock:
            placeholders = ",".join("?" for _ in uuids)
            cursor = self._conn.execute(
                f"SELECT local_event_uuid, payload FROM events "
                f"WHERE local_event_uuid IN ({placeholders})",
                uuids,
            )
            result = {}
            for row in cursor.fetchall():
                payload = json.loads(row[1])
                result[row[0]] = {
                    "query_text": payload.get("query_text"),
                    "response": payload.get("response"),
                }
            return result

    def mark_graded(self, uuid: str, reward: float) -> None:
        """Record a human grade locally."""
        with self._lock:
            self._conn.execute(
                "UPDATE events SET human_reward = ?, graded_at = ? "
                "WHERE local_event_uuid = ?",
                (reward, time.time(), uuid),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
