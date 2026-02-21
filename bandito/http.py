"""Sync HTTP transport — thin httpx.Client wrapper for cloud API."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("bandito")

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds — 0.5, 1.0, 2.0


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


class BanditoHTTP:
    """Sync HTTP client for cloud API communication.

    All methods return raw dicts (JSON). Auth via X-API-Key header.
    Retries transient errors (5xx, timeouts, connection errors) with
    exponential backoff. Never retries 4xx.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1",
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """Execute an HTTP request with retry logic."""
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.request(method, path, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    # Log full response body for client errors (4xx) — critical for debugging
                    # Pydantic 422 validation errors, auth failures, etc.
                    logger.warning(
                        "%s %s → %d: %s",
                        method, path, exc.response.status_code, exc.response.text,
                    )
                if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.debug(
                    "Retryable error on %s %s (attempt %d/%d), retrying in %.1fs: %s",
                    method, path, attempt + 1, MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
        raise last_exc  # unreachable, but satisfies type checker

    def connect(self) -> dict[str, Any]:
        """POST /sync/connect — SDK bootstrap."""
        return self._request("POST", "/sync/connect")

    def heartbeat(self, sdk_version: str | None = None) -> dict[str, Any]:
        """POST /sync/heartbeat — periodic state refresh."""
        body: dict[str, Any] = {}
        if sdk_version:
            body["sdk_version"] = sdk_version
        return self._request("POST", "/sync/heartbeat", json=body)

    def ingest_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """POST /events — batch event ingestion."""
        return self._request("POST", "/events", json={"events": events})

    def submit_grade(
        self, event_uuid: str, grade: float
    ) -> dict[str, Any]:
        """PATCH /events/{uuid}/grade — submit human grade."""
        return self._request(
            "PATCH",
            f"/events/{event_uuid}/grade",
            json={"grade": grade, "is_graded": True},
        )

    def list_bandits(self) -> dict[str, Any]:
        """GET /bandits — list all bandits for this user."""
        return self._request("GET", "/bandits")

    def close(self) -> None:
        self._client.close()
