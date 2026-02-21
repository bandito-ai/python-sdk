"""Tests for connect/sync state hydration."""

import numpy as np
import pytest
import httpx
import respx

from bandito.client import BanditoClient
from tests.conftest import ARM_DATA, EXPECTED_DIMS, make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


def _make_client(**kwargs) -> BanditoClient:
    return BanditoClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        store_path=":memory:",
        **kwargs,
    )


class TestConnect:
    @respx.mock
    def test_connect_hydrates_cache(self):
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        client = _make_client()
        client.connect()
        try:
            # Should be able to pull after connect
            result = client.pull("my-chatbot")
            assert result.bandit_name == "my-chatbot"
            assert result.bandit_id == 1
        finally:
            client.close()

    @respx.mock
    def test_connect_multiple_bandits(self):
        d = EXPECTED_DIMS
        sync_data = make_sync_response([
            {
                "bandit_id": 1, "name": "chatbot-a", "type": "online",
                "cost_importance": 0, "latency_importance": 0,
                "optimization_mode": "base", "total_pull_count": 0,
                "avg_latency_last_n": None,
                "theta": [0.0] * d, "cholesky": np.eye(d).tolist(),
                "dimensions": d,
                "arms": [{**a, "avg_latency_last_n": None} for a in ARM_DATA],
            },
            {
                "bandit_id": 2, "name": "chatbot-b", "type": "offline",
                "cost_importance": 1, "latency_importance": 1,
                "optimization_mode": "explore", "total_pull_count": 10,
                "avg_latency_last_n": 500.0,
                "theta": [0.1] * d, "cholesky": np.eye(d).tolist(),
                "dimensions": d,
                "arms": [{**a, "avg_latency_last_n": 500.0} for a in ARM_DATA],
            },
        ])
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=sync_data)
        )
        client = _make_client()
        client.connect()
        try:
            r1 = client.pull("chatbot-a")
            r2 = client.pull("chatbot-b")
            assert r1.bandit_id == 1
            assert r2.bandit_id == 2
        finally:
            client.close()

    @respx.mock
    def test_connect_skips_bandits_without_arms(self):
        sync_data = make_sync_response([{
            "bandit_id": 99, "name": "empty", "type": "online",
            "cost_importance": 0, "latency_importance": 0,
            "optimization_mode": "base", "total_pull_count": 0,
            "avg_latency_last_n": None,
            "theta": [], "cholesky": [],
            "dimensions": 0, "arms": [],
        }])
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=sync_data)
        )
        client = _make_client()
        client.connect()
        try:
            with pytest.raises(KeyError, match="empty"):
                client.pull("empty")
        finally:
            client.close()

    def test_connect_missing_api_key_raises(self, monkeypatch):
        from bandito.config import BanditoConfig
        monkeypatch.setattr(
            "bandito.config.load_config",
            lambda: BanditoConfig(),  # no api_key
        )
        monkeypatch.delenv("BANDITO_API_KEY", raising=False)
        client = BanditoClient()
        with pytest.raises(ValueError, match="api_key required"):
            client.connect()


class TestSync:
    @respx.mock
    def test_manual_sync_refreshes_state(self):
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        # Heartbeat returns updated theta
        d = EXPECTED_DIMS
        updated_sync = make_sync_response([{
            "bandit_id": 1, "name": "my-chatbot", "type": "online",
            "cost_importance": 2, "latency_importance": 3,
            "optimization_mode": "maximize",
            "total_pull_count": 100,
            "avg_latency_last_n": 1200.0,
            "theta": [0.5] * d, "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [{**a, "avg_latency_last_n": 1200.0} for a in ARM_DATA],
        }])
        respx.post(f"{BASE_URL}/api/v1/sync/heartbeat").mock(
            return_value=httpx.Response(200, json=updated_sync)
        )
        client = _make_client()
        client.connect()
        try:
            client.sync()
            # After sync, cache should have updated theta
            cache = client._bandits["my-chatbot"]
            assert cache.theta[0] == pytest.approx(0.5)
            assert cache.optimization_mode == "maximize"
        finally:
            client.close()

    @respx.mock
    def test_re_sync_replaces_bandits(self):
        """sync() replaces entire cache, not merge."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        # Heartbeat returns empty bandits
        respx.post(f"{BASE_URL}/api/v1/sync/heartbeat").mock(
            return_value=httpx.Response(200, json=make_sync_response([]))
        )
        client = _make_client()
        client.connect()
        try:
            assert "my-chatbot" in client._bandits
            client.sync()
            assert "my-chatbot" not in client._bandits
        finally:
            client.close()

    @respx.mock
    def test_arm_latency_parsed(self):
        d = EXPECTED_DIMS
        sync_data = make_sync_response([{
            "bandit_id": 1, "name": "test", "type": "online",
            "cost_importance": 0, "latency_importance": 0,
            "optimization_mode": "base", "total_pull_count": 0,
            "avg_latency_last_n": 1000.0,
            "theta": [0.0] * d, "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [
                {**ARM_DATA[0], "avg_latency_last_n": 800.0},
                {**ARM_DATA[1], "avg_latency_last_n": 1200.0},
                {**ARM_DATA[2], "avg_latency_last_n": None},
            ],
        }])
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=sync_data)
        )
        client = _make_client()
        client.connect()
        try:
            cache = client._bandits["test"]
            assert cache.arm_avg_latencies[1] == 800.0
            assert cache.arm_avg_latencies[2] == 1200.0
            assert cache.arm_avg_latencies[3] is None
        finally:
            client.close()

    @respx.mock
    def test_connect_with_pending_events_flushes(self):
        """Pending events from a previous crash get flushed on connect."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        ingest_route = respx.post(f"{BASE_URL}/api/v1/events").mock(
            return_value=httpx.Response(201, json={"accepted": 1, "duplicates": 0, "errors": []})
        )

        # Pre-populate SQLite with a pending event
        from bandito.store import EventStore
        store = EventStore(":memory:")
        store.push({
            "local_event_uuid": "leftover-1",
            "bandit_id": 1,
            "arm_id": 1,
            "query_text": "from crash",
        })

        client = _make_client()
        # inject pre-populated store before connect hydration
        from bandito.http import BanditoHTTP
        client._http = BanditoHTTP(BASE_URL, API_KEY)
        client._store = store
        data = client._http.connect()
        client._apply_sync(data)
        client._flush_pending()
        client._connected = True

        try:
            assert ingest_route.called
        finally:
            client._http.close()
            store.close()
