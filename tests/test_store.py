"""Tests for SQLite WAL event store."""

import pytest

from bandito.store import EventStore


def _make_event(uuid: str = "evt-1", bandit_id: int = 1, arm_id: int = 1):
    return {
        "local_event_uuid": uuid,
        "bandit_id": bandit_id,
        "arm_id": arm_id,
        "query_text": "hello",
    }


class TestEventStore:
    def test_push_and_pending(self):
        store = EventStore(":memory:")
        store.push(_make_event("a"))
        store.push(_make_event("b"))
        pending = store.pending()
        assert len(pending) == 2
        assert pending[0]["local_event_uuid"] == "a"
        store.close()

    def test_mark_flushed(self):
        store = EventStore(":memory:")
        store.push(_make_event("a"))
        store.push(_make_event("b"))
        store.mark_flushed(["a"])
        pending = store.pending()
        assert len(pending) == 1
        assert pending[0]["local_event_uuid"] == "b"
        store.close()

    def test_pending_limit(self):
        store = EventStore(":memory:")
        for i in range(10):
            store.push(_make_event(f"evt-{i}"))
        pending = store.pending(limit=3)
        assert len(pending) == 3
        store.close()

    def test_duplicate_uuid_ignored(self):
        store = EventStore(":memory:")
        store.push(_make_event("a"))
        store.push(_make_event("a"))  # duplicate
        pending = store.pending()
        assert len(pending) == 1
        store.close()

    def test_crash_recovery(self, tmp_path):
        """Events survive store close and reopen (simulates crash)."""
        db_path = str(tmp_path / "test.db")
        store = EventStore(db_path)
        store.push(_make_event("crash-1"))
        store.push(_make_event("crash-2"))
        store.close()

        # Reopen — events should still be pending
        store2 = EventStore(db_path)
        pending = store2.pending()
        assert len(pending) == 2
        store2.close()

    def test_mark_flushed_empty_list(self):
        store = EventStore(":memory:")
        store.mark_flushed([])  # should not raise
        store.close()
