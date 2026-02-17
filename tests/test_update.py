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
        flush_interval=9999,
        sync_interval=9999,
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
                response_text="world",
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
                response_text="r",
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
            assert event["immediate_reward"] == 0.5
            assert event["cost"] == 0.01
            assert event["latency"] == 500.0
            assert event["input_tokens"] == 100
            assert event["output_tokens"] == 200
            assert event["segment"] == {"tier": "pro"}
            assert event["query_text"] == "q"
            assert event["response_text"] == "r"
        finally:
            client.close()

    @respx.mock
    def test_update_optional_fields_omitted(self):
        """Fields not passed should not appear in payload."""
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            client.update(result)  # no optional fields
            event = client._store.pending()[0]
            assert "immediate_reward" not in event
            assert "cost" not in event
            assert "latency" not in event
            assert "query_text" not in event
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
            assert event["immediate_reward"] == 0.0
        finally:
            client.close()


class TestReward:
    @respx.mock
    def test_reward_sends_http_request(self):
        client = _connected_client()
        reward_route = respx.patch(f"{BASE_URL}/api/v1/events/evt-123/reward").mock(
            return_value=httpx.Response(200, json={
                "event_id": 1, "reward": 0.9,
                "computed_reward": 0.85, "is_human_reward": True,
                "state_updated": True,
            })
        )
        try:
            client.reward("evt-123", 0.9)
            assert reward_route.called
            request = reward_route.calls[0].request
            body = json.loads(request.content)
            assert body["reward"] == 0.9
            assert body["is_human_reward"] is True
        finally:
            client.close()

    @respx.mock
    def test_reward_machine_grade(self):
        client = _connected_client()
        reward_route = respx.patch(f"{BASE_URL}/api/v1/events/evt-456/reward").mock(
            return_value=httpx.Response(200, json={
                "event_id": 1, "reward": 0.7,
                "computed_reward": 0.65, "is_human_reward": False,
                "state_updated": True,
            })
        )
        try:
            client.reward("evt-456", 0.7, is_human=False)
            body = json.loads(reward_route.calls[0].request.content)
            assert body["is_human_reward"] is False
        finally:
            client.close()
