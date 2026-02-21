"""Tests for TUI API client."""

import pytest
import respx
from httpx import Response

from bandito.tui.api import TuiAPI

BASE = "http://localhost:8000/api/v1"


@pytest.fixture
def api():
    client = TuiAPI(api_key="bnd_test", base_url="http://localhost:8000")
    yield client
    client.close()


class TestListBandits:
    @respx.mock
    def test_success(self, api):
        payload = {"items": [{"id": 1, "name": "test"}], "total": 1}
        respx.get(f"{BASE}/bandits").mock(return_value=Response(200, json=payload))
        result = api.list_bandits()
        assert result["items"][0]["name"] == "test"

    @respx.mock
    def test_sends_api_key_header(self, api):
        route = respx.get(f"{BASE}/bandits").mock(
            return_value=Response(200, json={"items": [], "total": 0})
        )
        api.list_bandits()
        assert route.calls[0].request.headers["X-API-Key"] == "bnd_test"


class TestGetStats:
    @respx.mock
    def test_success(self, api):
        payload = {"bandit_id": 1, "total_events": 42}
        respx.get(f"{BASE}/analytics/1/stats").mock(
            return_value=Response(200, json=payload)
        )
        result = api.get_stats(1)
        assert result["total_events"] == 42


class TestGetArmPerformance:
    @respx.mock
    def test_success(self, api):
        payload = {"bandit_id": 1, "arms": [{"arm_id": 1, "pull_share": 0.6}]}
        respx.get(f"{BASE}/analytics/1/arms/performance").mock(
            return_value=Response(200, json=payload)
        )
        result = api.get_arm_performance(1)
        assert result["arms"][0]["pull_share"] == 0.6


class TestListEvents:
    @respx.mock
    def test_filters_by_bandit(self, api):
        route = respx.get(f"{BASE}/events").mock(
            return_value=Response(200, json={"items": [], "total": 0})
        )
        api.list_events(1, has_grade=False)
        params = dict(route.calls[0].request.url.params)
        assert params["bandit_id"] == "1"
        assert params["has_grade"] == "false"


class TestSubmitGrade:
    @respx.mock
    def test_sends_grade(self, api):
        route = respx.patch(f"{BASE}/events/uuid-123/grade").mock(
            return_value=Response(200, json={"event_id": 1, "grade": 1.0})
        )
        api.submit_grade("uuid-123", 1.0)
        body = route.calls[0].request.content
        import json
        parsed = json.loads(body)
        assert parsed["grade"] == 1.0
        assert parsed["is_graded"] is True
