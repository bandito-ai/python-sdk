"""Integration tests — full connect -> pull -> update -> reward cycle."""

import json

import httpx
import pytest
import respx

from bandito._worker import prepare_cloud_payload
from bandito.client import BanditoClient
from tests.conftest import make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


class TestFullCycle:
    @respx.mock
    def test_connect_pull_update_reward(self):
        """Full lifecycle: connect -> pull -> update -> reward."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY,
            base_url=BASE_URL,
            store_path=":memory:",
            data_storage="cloud",
        )
        client.connect()

        try:
            # Pull
            result = client.pull("my-chatbot", query="What is AI?")
            assert result.model in {"gpt-4", "claude-sonnet"}

            # Update
            client.update(
                result,
                query_text="What is AI?",
                response="AI is artificial intelligence.",
                reward=0.85,
                cost=0.003,
                latency=1200.0,
            )

            # Drain executor so flush completes before assertions
            client._executor.shutdown(wait=True)
            client._executor = None

            assert ingest_route.called

            # Verify ingest payload
            body = json.loads(ingest_route.calls[0].request.content)
            assert len(body["events"]) == 1
            event = body["events"][0]
            assert event["local_event_uuid"] == result.event_id
            assert event["query_text"] == "What is AI?"

            # Delayed grade
            grade_route = respx.patch(
                f"{BASE_URL}/api/v1/events/{result.event_id}/grade"
            ).mock(
                return_value=httpx.Response(200, json={
                    "event_id": 1, "grade": 0.9,
                    "reward": 0.88, "is_graded": True,
                    "state_updated": True,
                })
            )
            client.grade(result.event_id, 0.9)
            assert grade_route.called
        finally:
            client.close()

    @respx.mock
    def test_close_flushes_remaining_events(self):
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            store_path=":memory:",
        )
        client.connect()
        result = client.pull("my-chatbot")
        client.update(result, reward=0.5)

        # Close should drain executor + final flush
        client.close()
        assert ingest_route.called

    @respx.mock
    def test_close_survives_http_error(self):
        """close() should not raise even if final flush fails."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(500, json={"detail": "server error"})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            store_path=":memory:",
        )
        client.connect()
        result = client.pull("my-chatbot")
        client.update(result, reward=0.5)

        # Should not raise
        client.close()

    @respx.mock
    def test_multiple_pulls_different_arms(self):
        """Multiple pulls should sometimes select different arms (stochastic)."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            store_path=":memory:",
        )
        client.connect()
        try:
            arm_ids = set()
            for _ in range(50):
                result = client.pull("my-chatbot")
                arm_ids.add(result.arm.arm_id)
            # With cold-start (identity chol, zero theta), Thompson Sampling
            # is purely random — should hit at least 2 different arms in 50 tries
            assert len(arm_ids) >= 2
        finally:
            client.close()

    @respx.mock
    def test_module_level_api(self):
        """Test the import bandito; bandito.connect() pattern."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        import bandito
        bandito.connect(
            api_key=API_KEY,
            base_url=BASE_URL,
            store_path=":memory:",
        )
        try:
            result = bandito.pull("my-chatbot")
            assert result.bandit_name == "my-chatbot"
        finally:
            bandito.close()

    @respx.mock
    def test_context_manager(self):
        """with BanditoClient(...) as client: connects and exits cleanly."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        with BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:",
        ) as client:
            result = client.pull("my-chatbot")
            assert result.bandit_name == "my-chatbot"

        # After exiting, client should be closed
        assert not client._connected

    @respx.mock
    def test_update_survives_http_failure(self):
        """update() doesn't raise when flush fails; event stays pending."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(500, json={"detail": "server error"})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:",
        )
        client.connect()
        try:
            result = client.pull("my-chatbot")
            # Should not raise even though flush will fail
            client.update(result, reward=0.5)
            # Drain executor to ensure flush attempt completes
            client._executor.shutdown(wait=True)
            client._executor = None
            # Event should still be pending (flush failed)
            assert len(client._store.pending()) == 1
        finally:
            client.close()


class TestPrepareCloudPayload:
    def test_strips_text_and_metadata_when_local(self):
        events = [
            {"local_event_uuid": "a", "query_text": "hi", "response": "hello",
             "model_name": "gpt-4", "model_provider": "openai", "cost": 0.1},
            {"local_event_uuid": "b", "query_text": "bye", "model_name": "claude"},
        ]
        stripped = prepare_cloud_payload(events, include_text=False)
        assert len(stripped) == 2
        assert "query_text" not in stripped[0]
        assert "response" not in stripped[0]
        assert "model_name" not in stripped[0]
        assert "model_provider" not in stripped[0]
        assert stripped[0]["cost"] == 0.1
        assert "query_text" not in stripped[1]
        assert "model_name" not in stripped[1]
        # Originals unchanged
        assert events[0]["query_text"] == "hi"
        assert events[0]["model_name"] == "gpt-4"

    def test_keeps_text_but_strips_metadata_when_cloud(self):
        events = [
            {"local_event_uuid": "a", "query_text": "hi", "response": "hello",
             "model_name": "gpt-4", "model_provider": "openai", "cost": 0.1},
        ]
        stripped = prepare_cloud_payload(events, include_text=True)
        assert stripped[0]["query_text"] == "hi"
        assert stripped[0]["response"] == "hello"
        assert "model_name" not in stripped[0]
        assert "model_provider" not in stripped[0]

    def test_no_optional_fields_is_noop(self):
        events = [{"local_event_uuid": "a", "cost": 0.1}]
        stripped = prepare_cloud_payload(events, include_text=False)
        assert stripped == events

    @respx.mock
    def test_local_storage_strips_text_on_flush(self):
        """With data_storage='local', flushed payload omits text and metadata fields."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            store_path=":memory:", data_storage="local",
        )
        client.connect()
        try:
            result = client.pull("my-chatbot")
            client.update(
                result,
                query_text="What is AI?",
                response="AI is artificial intelligence.",
            )

            # Drain executor so flush completes
            client._executor.shutdown(wait=True)
            client._executor = None

            # SQLite should have had the full text + metadata before flush
            # After flush, events are marked flushed so pending() is empty.
            # Check the HTTP call instead.
            assert ingest_route.called
            body = json.loads(ingest_route.calls[0].request.content)
            event = body["events"][0]
            assert "query_text" not in event
            assert "response" not in event
            assert "model_name" not in event
            assert "model_provider" not in event
        finally:
            client.close()

    @respx.mock
    def test_cloud_storage_keeps_text_on_flush(self):
        """With data_storage='cloud', flushed payload includes text but strips metadata."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            store_path=":memory:", data_storage="cloud",
        )
        client.connect()
        try:
            result = client.pull("my-chatbot")
            client.update(
                result,
                query_text="What is AI?",
                response="AI is artificial intelligence.",
            )

            # Drain executor so flush completes
            client._executor.shutdown(wait=True)
            client._executor = None

            assert ingest_route.called
            body = json.loads(ingest_route.calls[0].request.content)
            event = body["events"][0]
            assert event["query_text"] == "What is AI?"
            assert event["response"] == {"response": "AI is artificial intelligence."}
            assert "model_name" not in event
            assert "model_provider" not in event
        finally:
            client.close()


class TestPartialAcceptance:
    """Tests for partial acceptance and poison pill handling in _flush_pending."""

    @respx.mock
    def test_partial_acceptance_flushes_good_events(self):
        """Good events are marked flushed even when one event errors."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:",
        )
        client.connect()
        try:
            r1 = client.pull("my-chatbot")
            r2 = client.pull("my-chatbot")
            # Shut down executor so we control flush timing
            client._executor.shutdown(wait=True)
            client._executor = None
            client._store.push({
                "local_event_uuid": r1.event_id,
                "bandit_id": r1.bandit_id,
                "arm_id": r1.arm.arm_id,
            })
            client._store.push({
                "local_event_uuid": r2.event_id,
                "bandit_id": r2.bandit_id,
                "arm_id": r2.arm.arm_id,
            })

            # Server accepts r1 but rejects r2
            respx.post(f"{BASE_URL}/api/v1/events").mock(
                return_value=httpx.Response(201, json={
                    "accepted": 1, "duplicates": 0,
                    "errors": [{"local_event_uuid": r2.event_id, "reason": "bad data"}],
                })
            )

            client._flush_pending()

            # r1 should be flushed, r2 should still be pending
            pending = client._store.pending()
            assert len(pending) == 1
            assert pending[0]["local_event_uuid"] == r2.event_id
        finally:
            client.close()

    @respx.mock
    def test_poison_pill_becomes_dead_after_max_retries(self):
        """An event rejected 5 times is skipped on subsequent flushes."""
        from bandito.client import _MAX_EVENT_RETRIES

        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:",
        )
        client.connect()
        try:
            r1 = client.pull("my-chatbot")
            # Shut down executor so we control flush timing
            client._executor.shutdown(wait=True)
            client._executor = None
            client._store.push({
                "local_event_uuid": r1.event_id,
                "bandit_id": r1.bandit_id,
                "arm_id": r1.arm.arm_id,
            })

            # Server rejects this event every time
            ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
                return_value=httpx.Response(201, json={
                    "accepted": 0, "duplicates": 0,
                    "errors": [{"local_event_uuid": r1.event_id, "reason": "permanently bad"}],
                })
            )

            # Flush _MAX_EVENT_RETRIES times
            for i in range(_MAX_EVENT_RETRIES):
                client._flush_pending()
                assert r1.event_id not in client._dead_uuids or i == _MAX_EVENT_RETRIES - 1

            # Now it's dead
            assert r1.event_id in client._dead_uuids

            # Next flush should skip it entirely (no HTTP call)
            ingest_route.calls.clear()
            client._flush_pending()
            assert not ingest_route.called
        finally:
            client.close()

    @respx.mock
    def test_dead_events_reset_on_reconnect(self):
        """Reconnecting clears dead set, giving events another chance."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:",
        )
        client.connect()
        try:
            # Simulate a dead event
            client._dead_uuids.add("some-uuid")
            client._retry_counts["some-uuid"] = 5
            assert len(client._dead_uuids) == 1

            # Reconnect
            client.connect()
            assert len(client._dead_uuids) == 0
            assert len(client._retry_counts) == 0
        finally:
            client.close()

    @respx.mock
    def test_good_events_flush_past_poison_pill(self):
        """New events flush successfully even when a dead event exists."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:",
        )
        client.connect()
        try:
            client._executor.shutdown(wait=True)
            client._executor = None

            # Simulate a dead event already in the store
            client._store.push({
                "local_event_uuid": "bad-event",
                "bandit_id": 1,
                "arm_id": 1,
            })
            client._dead_uuids.add("bad-event")

            # Push a good event
            r1 = client.pull("my-chatbot")
            client._store.push({
                "local_event_uuid": r1.event_id,
                "bandit_id": r1.bandit_id,
                "arm_id": r1.arm.arm_id,
            })

            ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
                return_value=httpx.Response(201, json={
                    "accepted": 1, "duplicates": 0, "errors": [],
                })
            )

            client._flush_pending()

            # Good event flushed, bad event still pending but skipped
            assert ingest_route.called
            body = json.loads(ingest_route.calls[0].request.content)
            sent_uuids = [e["local_event_uuid"] for e in body["events"]]
            assert r1.event_id in sent_uuids
            assert "bad-event" not in sent_uuids
        finally:
            client.close()


class TestEventQuotaWarning:
    @respx.mock
    def test_flush_logs_quota_warning(self, caplog):
        """Ingest response with warning field logs it via logger.warning."""
        import logging

        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        warning_msg = "Event quota 90% used (900/1000 this month)"
        respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={
                "accepted": 1, "duplicates": 0, "errors": [],
                "warning": warning_msg,
            })
        )

        client = BanditoClient(
            api_key=API_KEY,
            base_url=BASE_URL,
            store_path=":memory:",
        )
        try:
            client.connect()
            result = client.pull("my-chatbot", query="hello")
            client.update(result, response="world", reward=0.8)

            with caplog.at_level(logging.WARNING, logger="bandito.client"):
                client._flush_pending()

            assert warning_msg in caplog.text
        finally:
            client.close()
