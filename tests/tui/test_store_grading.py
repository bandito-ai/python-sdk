"""Tests for EventStore grading extensions."""

import json
import time

import pytest

from bandito.store import EventStore


def _make_event(uuid: str, bandit_id: int = 1, arm_id: int = 1) -> dict:
    return {
        "local_event_uuid": uuid,
        "bandit_id": bandit_id,
        "arm_id": arm_id,
        "query_text": f"query for {uuid}",
        "response_text": f"response for {uuid}",
        "model_name": "gpt-4",
        "model_provider": "openai",
    }


@pytest.fixture
def store():
    s = EventStore(":memory:")
    yield s
    s.close()


class TestUngraded:
    def test_returns_flushed_ungraded(self, store):
        store.push(_make_event("aaa"))
        store.push(_make_event("bbb"))
        store.mark_flushed(["aaa", "bbb"])
        result = store.ungraded()
        assert len(result) == 2

    def test_excludes_pending(self, store):
        store.push(_make_event("pending-1"))
        result = store.ungraded()
        assert len(result) == 0

    def test_excludes_graded(self, store):
        store.push(_make_event("graded-1"))
        store.mark_flushed(["graded-1"])
        store.mark_graded("graded-1", 1.0)
        result = store.ungraded()
        assert len(result) == 0

    def test_filters_by_bandit_id(self, store):
        store.push(_make_event("b1", bandit_id=1))
        store.push(_make_event("b2", bandit_id=2))
        store.mark_flushed(["b1", "b2"])
        result = store.ungraded(bandit_id=1)
        assert len(result) == 1
        assert result[0]["local_event_uuid"] == "b1"

    def test_respects_limit(self, store):
        for i in range(10):
            store.push(_make_event(f"ev-{i}"))
        store.mark_flushed([f"ev-{i}" for i in range(10)])
        result = store.ungraded(limit=3)
        assert len(result) == 3


class TestMarkGraded:
    def test_sets_human_reward_and_graded_at(self, store):
        store.push(_make_event("test-grade"))
        store.mark_flushed(["test-grade"])
        store.mark_graded("test-grade", 0.0)

        # Verify via direct SQL
        cursor = store._conn.execute(
            "SELECT human_reward, graded_at FROM events WHERE local_event_uuid = ?",
            ("test-grade",),
        )
        row = cursor.fetchone()
        assert row[0] == 0.0
        assert row[1] is not None
        assert row[1] > 0


class TestMigration:
    def test_migration_idempotent(self):
        """Calling __init__ twice doesn't break (migration is safe to re-run)."""
        store = EventStore(":memory:")
        # Manually run migration again
        store._migrate()
        store.push(_make_event("migrated"))
        store.mark_flushed(["migrated"])
        store.mark_graded("migrated", 1.0)
        result = store.ungraded()
        assert len(result) == 0
        store.close()
