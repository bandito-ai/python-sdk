"""Integration tests — full connect → pull → update → reward cycle."""

import json

import httpx
import pytest
import respx

from bandito._worker import strip_text_fields
from bandito.client import BanditoClient
from tests.conftest import make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


class TestFullCycle:
    @respx.mock
    def test_connect_pull_update_reward(self):
        """Full lifecycle: connect → pull → update → reward."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY,
            base_url=BASE_URL,
            flush_interval=9999,
            sync_interval=9999,
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
                response_text="AI is artificial intelligence.",
                reward=0.85,
                cost=0.003,
                latency=1200.0,
            )

            # Verify event in SQLite
            pending = client._store.pending()
            assert len(pending) == 1
            assert pending[0]["immediate_reward"] == 0.85

            # Manually flush (simulate what worker does)
            client._flush_pending()
            assert ingest_route.called

            # Verify ingest payload
            body = json.loads(ingest_route.calls[0].request.content)
            assert len(body["events"]) == 1
            event = body["events"][0]
            assert event["local_event_uuid"] == result.event_id
            assert event["query_text"] == "What is AI?"

            # Delayed reward
            reward_route = respx.patch(
                f"{BASE_URL}/api/v1/events/{result.event_id}/reward"
            ).mock(
                return_value=httpx.Response(200, json={
                    "event_id": 1, "reward": 0.9,
                    "computed_reward": 0.88, "is_human_reward": True,
                    "state_updated": True,
                })
            )
            client.reward(result.event_id, 0.9)
            assert reward_route.called
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
            flush_interval=9999, sync_interval=9999,
            store_path=":memory:",
        )
        client.connect()
        result = client.pull("my-chatbot")
        client.update(result, reward=0.5)

        # Close should attempt final flush
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
            flush_interval=9999, sync_interval=9999,
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
            flush_interval=9999, sync_interval=9999,
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
            flush_interval=9999,
            sync_interval=9999,
            store_path=":memory:",
        )
        try:
            result = bandito.pull("my-chatbot")
            assert result.bandit_name == "my-chatbot"
        finally:
            bandito.close()


class TestStripTextFields:
    def test_strips_text_fields(self):
        events = [
            {"local_event_uuid": "a", "query_text": "hi", "response_text": "hello", "cost": 0.1},
            {"local_event_uuid": "b", "query_text": "bye"},
        ]
        stripped = strip_text_fields(events)
        assert len(stripped) == 2
        assert "query_text" not in stripped[0]
        assert "response_text" not in stripped[0]
        assert stripped[0]["cost"] == 0.1
        assert "query_text" not in stripped[1]
        # Originals unchanged
        assert events[0]["query_text"] == "hi"

    def test_no_text_fields_is_noop(self):
        events = [{"local_event_uuid": "a", "cost": 0.1}]
        stripped = strip_text_fields(events)
        assert stripped == events

    @respx.mock
    def test_local_storage_strips_text_on_flush(self):
        """With data_storage='local', flushed payload omits text fields."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            flush_interval=9999, sync_interval=9999,
            store_path=":memory:", data_storage="local",
        )
        client.connect()
        try:
            result = client.pull("my-chatbot")
            client.update(
                result,
                query_text="What is AI?",
                response_text="AI is artificial intelligence.",
            )

            # SQLite should have the full text
            pending = client._store.pending()
            assert pending[0]["query_text"] == "What is AI?"
            assert pending[0]["response_text"] == "AI is artificial intelligence."

            # Flush to cloud
            client._flush_pending()
            assert ingest_route.called
            body = json.loads(ingest_route.calls[0].request.content)
            event = body["events"][0]
            assert "query_text" not in event
            assert "response_text" not in event
        finally:
            client.close()

    @respx.mock
    def test_cloud_storage_keeps_text_on_flush(self):
        """With data_storage='cloud', flushed payload includes text fields."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL,
            flush_interval=9999, sync_interval=9999,
            store_path=":memory:", data_storage="cloud",
        )
        client.connect()
        try:
            result = client.pull("my-chatbot")
            client.update(
                result,
                query_text="What is AI?",
                response_text="AI is artificial intelligence.",
            )

            client._flush_pending()
            assert ingest_route.called
            body = json.loads(ingest_route.calls[0].request.content)
            event = body["events"][0]
            assert event["query_text"] == "What is AI?"
            assert event["response_text"] == "AI is artificial intelligence."
        finally:
            client.close()
