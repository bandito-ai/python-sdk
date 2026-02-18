"""Tests for budget warning logging."""

import logging

import httpx
import pytest
import respx

from bandito.client import BanditoClient
from tests.conftest import make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


def _make_client() -> BanditoClient:
    return BanditoClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        store_path=":memory:",
    )


class TestBudgetWarnings:
    @respx.mock
    def test_no_warning_no_budget(self, caplog):
        """No warning when budget is None."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(200, json=make_sync_response())
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "budget" not in caplog.text.lower() or "approaching" not in caplog.text
        finally:
            client.close()

    @respx.mock
    def test_no_warning_below_90(self, caplog):
        """No warning when spend is below 90% of budget."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(
                200, json=make_sync_response(budget=100.0, total_cost=80.0)
            )
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "approaching" not in caplog.text
            assert "reached" not in caplog.text
            assert "EXCEEDED" not in caplog.text
        finally:
            client.close()

    @respx.mock
    def test_warning_approaching(self, caplog):
        """Warning logged when spend is 90-99% of budget."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(
                200, json=make_sync_response(budget=100.0, total_cost=92.0)
            )
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "approaching" in caplog.text
        finally:
            client.close()

    @respx.mock
    def test_warning_reached(self, caplog):
        """Warning logged when spend is 100-109% of budget."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(
                200, json=make_sync_response(budget=100.0, total_cost=105.0)
            )
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "reached" in caplog.text
        finally:
            client.close()

    @respx.mock
    def test_warning_exceeded(self, caplog):
        """Warning logged when spend is 110%+ of budget."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(
                200, json=make_sync_response(budget=100.0, total_cost=115.0)
            )
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "EXCEEDED" in caplog.text
        finally:
            client.close()

    @respx.mock
    def test_no_warning_zero_budget(self, caplog):
        """No warning when budget is zero (avoids division by zero)."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(
                200, json=make_sync_response(budget=0.0, total_cost=5.0)
            )
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "budget" not in caplog.text.lower() or "approaching" not in caplog.text
        finally:
            client.close()

    @respx.mock
    def test_no_warning_null_total_cost(self, caplog):
        """No warning when total_cost is None (no events yet)."""
        respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
            return_value=httpx.Response(
                200, json=make_sync_response(budget=100.0, total_cost=None)
            )
        )
        client = _make_client()
        with caplog.at_level(logging.WARNING, logger="bandito"):
            client.connect()
        try:
            assert "approaching" not in caplog.text
            assert "reached" not in caplog.text
            assert "EXCEEDED" not in caplog.text
        finally:
            client.close()
