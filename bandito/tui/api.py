"""TUI HTTP client — analytics, bandits, and reward submission."""

from __future__ import annotations

from typing import Any

import httpx


class TuiAPI:
    """Sync HTTP client for TUI ↔ cloud communication.

    Follows the same pattern as sdk/bandito/http.py but exposes
    endpoints the TUI needs (analytics, bandit listing, reward).
    """

    def __init__(self, api_key: str, base_url: str, timeout: float = 10.0):
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/api/v1",
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    # ── Bandits ─────────────────────────────────────────────────────

    def list_bandits(self) -> dict[str, Any]:
        """GET /bandits — paginated bandit list."""
        resp = self._client.get("/bandits", params={"limit": 100})
        resp.raise_for_status()
        return resp.json()

    def get_bandit(self, bandit_id: int) -> dict[str, Any]:
        """GET /bandits/{id} — single bandit detail."""
        resp = self._client.get(f"/bandits/{bandit_id}")
        resp.raise_for_status()
        return resp.json()

    # ── Analytics ───────────────────────────────────────────────────

    def get_stats(self, bandit_id: int) -> dict[str, Any]:
        """GET /analytics/{id}/stats — bandit-level stats."""
        resp = self._client.get(f"/analytics/{bandit_id}/stats")
        resp.raise_for_status()
        return resp.json()

    def get_arm_performance(self, bandit_id: int) -> dict[str, Any]:
        """GET /analytics/{id}/arms/performance — per-arm metrics."""
        resp = self._client.get(f"/analytics/{bandit_id}/arms/performance")
        resp.raise_for_status()
        return resp.json()

    # ── Events ──────────────────────────────────────────────────────

    def list_events(
        self,
        bandit_id: int,
        *,
        has_human_reward: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """GET /events — paginated event list."""
        params: dict[str, Any] = {
            "bandit_id": bandit_id,
            "limit": limit,
            "offset": offset,
        }
        if has_human_reward is not None:
            params["has_human_reward"] = has_human_reward
        resp = self._client.get("/events", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_event(self, event_uuid: str) -> dict[str, Any]:
        """GET /events/{uuid} — full event detail."""
        resp = self._client.get(f"/events/{event_uuid}")
        resp.raise_for_status()
        return resp.json()

    def submit_grade(self, event_uuid: str, reward: float) -> dict[str, Any]:
        """PATCH /events/{uuid}/reward — submit human grade."""
        resp = self._client.patch(
            f"/events/{event_uuid}/reward",
            json={"reward": reward, "is_human_reward": True},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()
