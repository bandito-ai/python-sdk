"""Tests for update() — event payload shape and SQLite write."""

import json

import httpx
import pytest
import respx

from bandito.client import BanditoClient
from bandito.store import EventStore
from tests.conftest import ARM_DATA, EXPECTED_DIMS, make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


def _connected_client() -> BanditoClient:
    respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
        return_value=httpx.Response(200, json=make_sync_response())
    )
    client = BanditoClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        store_path=":memory:",
    )
    client.connect()
    return client


class TestUpdate:
    @respx.mock
    def test_update_writes_to_store(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(
                result,
                query_text="hello",
                response="world",
                reward=0.85,
                cost=0.003,
                latency=1200.0,
            )
            pending = client._store.pending()
            assert len(pending) == 1
            event = pending[0]
            assert event["local_event_uuid"] == result.event_id
            assert event["bandit_id"] == result.bandit_id
            assert event["arm_id"] == result.arm.arm_id
        finally:
            client.close()

    @respx.mock
    def test_update_payload_matches_event_ingest_schema(self):
        """Verify payload field names match backend EventIngest exactly."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(
                result,
                query_text="q",
                response="r",
                reward=0.5,
                cost=0.01,
                latency=500.0,
                input_tokens=100,
                output_tokens=200,
                segment={"tier": "pro"},
            )
            event = client._store.pending()[0]
            # Required fields
            assert "local_event_uuid" in event
            assert "bandit_id" in event
            assert "arm_id" in event
            # Optional fields match backend names
            assert event["early_reward"] == 0.5
            assert event["cost"] == 0.01
            assert event["latency"] == 500.0
            assert event["input_tokens"] == 100
            assert event["output_tokens"] == 200
            assert event["segment"] == {"tier": "pro"}
            assert event["query_text"] == "q"
            assert event["response"] == {"response": "r"}
        finally:
            client.close()

    @respx.mock
    def test_update_optional_fields_omitted(self):
        """Fields not passed should not appear in payload (except auto-latency)."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result)  # no optional fields
            event = client._store.pending()[0]
            assert "early_reward" not in event
            assert "cost" not in event
            assert "query_text" not in event
            # latency is auto-calculated from pull() timestamp
            assert "latency" in event
            assert event["latency"] > 0
        finally:
            client.close()

    @respx.mock
    def test_update_not_connected_raises(self):
        client = BanditoClient(api_key="x")
        from bandito.models import Arm, PullResult
        fake_result = PullResult(
            arm=Arm(arm_id=1, model_name="x", model_provider="x",
                    system_prompt="x", is_prompt_templated=False),
            event_id="x", bandit_id=1, bandit_name="x", scores={},
        )
        with pytest.raises(RuntimeError, match="Not connected"):
            client.update(fake_result)

    @respx.mock
    def test_update_multiple_events(self):
        client = _connected_client()
        try:
            r1 = client.pull("my-chatbot")
            r2 = client.pull("my-chatbot")
            client.update(r1, reward=0.5)
            client.update(r2, reward=0.9)
            pending = client._store.pending()
            assert len(pending) == 2
            uuids = {e["local_event_uuid"] for e in pending}
            assert r1.event_id in uuids
            assert r2.event_id in uuids
        finally:
            client.close()

    @respx.mock
    def test_update_reward_zero(self):
        """Edge case: reward=0.0 should still be included (not treated as falsy)."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result, reward=0.0)
            event = client._store.pending()[0]
            assert event["early_reward"] == 0.0
        finally:
            client.close()


class TestResponseTextNormalization:
    """Verify response is normalized to dict before storage."""

    @respx.mock
    def test_string_normalized_to_dict(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result, response="hello")
            event = client._store.pending()[0]
            assert event["response"] == {"response": "hello"}
        finally:
            client.close()

    @respx.mock
    def test_dict_stored_as_is(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            rich = {"choices": [{"text": "Hi"}], "usage": {"tokens": 5}}
            client.update(result, response=rich)
            event = client._store.pending()[0]
            assert event["response"] == rich
        finally:
            client.close()

    @respx.mock
    def test_none_omitted(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result)
            event = client._store.pending()[0]
            assert "response" not in event
        finally:
            client.close()

    @respx.mock
    def test_cloud_payload_preserves_dict(self):
        """Verify the dict structure survives through to the cloud payload."""
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={
                "accepted": 1, "duplicates": 0, "errors": [],
            })
        )
        client = _connected_client()
        client._data_storage = "cloud"
        try:
            result = client.pull("my-chatbot")
            client.update(result, response="hello")
            client._executor.shutdown(wait=True)
            client._executor = None

            body = json.loads(ingest_route.calls[0].request.content)
            event = body["events"][0]
            assert event["response"] == {"response": "hello"}
        finally:
            client.close()


class TestGrade:
    @respx.mock
    def test_grade_sends_http_request(self):
        client = _connected_client()
        grade_route = respx.patch(f"{BASE_URL}/api/v1/events/evt-123/grade").mock(
            return_value=httpx.Response(200, json={
                "event_id": 1, "grade": 0.9,
                "reward": 0.85, "is_graded": True,
                "state_updated": True,
            })
        )
        try:
            client.grade("evt-123", 0.9)
            assert grade_route.called
            request = grade_route.calls[0].request
            body = json.loads(request.content)
            assert body["grade"] == 0.9
            assert body["is_graded"] is True
        finally:
            client.close()


class TestFailedUpdate:
    """Tests for the failed=True update path."""

    @respx.mock
    def test_failed_sets_reward_zero(self):
        """failed=True with no reward defaults early_reward to 0.0."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result, failed=True)
            event = client._store.pending()[0]
            assert event["early_reward"] == 0.0
        finally:
            client.close()

    @respx.mock
    def test_failed_preserves_explicit_reward(self):
        """failed=True with explicit reward keeps the explicit value."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result, failed=True, reward=0.1)
            event = client._store.pending()[0]
            assert event["early_reward"] == 0.1
        finally:
            client.close()

    @respx.mock
    def test_failed_sets_run_error_flag(self):
        """failed=True adds run_error: True to the event."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result, failed=True)
            event = client._store.pending()[0]
            assert event["run_error"] is True
        finally:
            client.close()

    @respx.mock
    def test_failed_auto_latency_still_calculated(self):
        """Latency is auto-calculated even on failure events."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result, failed=True)
            event = client._store.pending()[0]
            assert "latency" in event
            assert event["latency"] > 0
        finally:
            client.close()

    @respx.mock
    def test_failed_false_no_effect(self):
        """Default failed=False has no run_error key and no reward default."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result)
            event = client._store.pending()[0]
            assert "run_error" not in event
            assert "early_reward" not in event
        finally:
            client.close()
