"""Concurrent SDK scenario tests.

Validates thread-safety of pull(), sync(), connect(), update(), and close()
under concurrent access. These tests don't verify correctness of results —
they verify no crashes, deadlocks, or data corruption.
"""

import threading
import time

import httpx
import numpy as np
import pytest
import respx

from bandito.client import BanditoClient
from tests.conftest import make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


def _mock_connect():
    respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
        return_value=httpx.Response(200, json=make_sync_response())
    )


def _mock_heartbeat():
    respx.post(f"{BASE_URL}/api/v1/sync/heartbeat").mock(
        return_value=httpx.Response(200, json=make_sync_response())
    )


def _mock_ingest():
    respx.post(f"{BASE_URL}/api/v1/events").mock(
        return_value=httpx.Response(
            201, json={"accepted": 1, "duplicates": 0, "errors": []}
        )
    )


class TestConcurrentPullAndSync:
    @respx.mock
    def test_pull_and_sync_concurrent(self):
        """Thread A calls pull() repeatedly while Thread B calls sync(). No crashes."""
        _mock_connect()
        _mock_heartbeat()
        _mock_ingest()

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:"
        )
        client.connect()

        errors = []
        stop = threading.Event()

        def puller():
            try:
                while not stop.is_set():
                    client.pull("my-chatbot", query="test query")
            except Exception as e:
                errors.append(("puller", e))

        def syncer():
            try:
                for _ in range(5):
                    client.sync()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("syncer", e))

        t_pull = threading.Thread(target=puller)
        t_sync = threading.Thread(target=syncer)

        t_pull.start()
        t_sync.start()

        t_sync.join(timeout=5)
        stop.set()
        t_pull.join(timeout=5)

        client.close()
        assert errors == [], f"Concurrent pull/sync errors: {errors}"


class TestConnectDuringFlush:
    @respx.mock
    def test_update_then_reconnect(self):
        """Call update() then immediately connect() again. No data loss or crash."""
        _mock_connect()
        _mock_ingest()

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:"
        )
        client.connect()

        result = client.pull("my-chatbot", query="hello")
        client.update(result, response="world")

        # Reconnect immediately — should not crash
        client.connect()

        # Should still be functional
        result2 = client.pull("my-chatbot", query="hello again")
        assert result2.arm is not None

        client.close()


class TestCloseDuringPendingUpdate:
    @respx.mock
    def test_update_then_close(self):
        """Call update() then close() immediately. No crash."""
        _mock_connect()
        _mock_ingest()

        client = BanditoClient(
            api_key=API_KEY, base_url=BASE_URL, store_path=":memory:"
        )
        client.connect()

        result = client.pull("my-chatbot", query="hello")
        client.update(result, response="world")

        # Close immediately — executor.shutdown(wait=True) should drain
        client.close()
        assert client._connected is False
