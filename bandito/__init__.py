"""Bandito SDK — contextual bandit optimization for LLM selection.

Usage:
    import bandito

    bandito.connect()  # reads API key from ~/.bandito/config.toml or BANDITO_API_KEY
    result = bandito.pull("my-chatbot", query=user_message)
    response = openai.chat.completions.create(
        model=result.model,
        messages=[
            {"role": "system", "content": result.prompt},
            {"role": "user", "content": user_message},
        ],
    )
    bandito.update(result, query_text=user_message, response_text=response_text)
"""

import threading

from bandito.client import BanditoClient
from bandito.models import Arm, PullResult

__all__ = [
    "BanditoClient",
    "Arm",
    "PullResult",
    "connect",
    "pull",
    "update",
    "reward",
    "sync",
    "close",
]

_client: BanditoClient | None = None
_lock = threading.Lock()


def _get_client() -> BanditoClient:
    if _client is None:
        raise RuntimeError("Not connected — call bandito.connect() first")
    return _client


def connect(api_key: str | None = None, **kwargs) -> None:
    """Connect to the Bandito cloud and hydrate local state."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
        _client = BanditoClient(api_key=api_key, **kwargs)
        _client.connect()


def pull(bandit_name: str, **kwargs) -> PullResult:
    """Local Thompson Sampling decision. <1ms, no network."""
    return _get_client().pull(bandit_name, **kwargs)


def update(pull_result: PullResult, **kwargs) -> None:
    """Send event data to cloud (writes to SQLite first)."""
    _get_client().update(pull_result, **kwargs)


def reward(event_id: str, reward: float, **kwargs) -> None:
    """Send a delayed reward for an existing event."""
    _get_client().reward(event_id, reward, **kwargs)


def sync() -> None:
    """Explicit state refresh from cloud."""
    _get_client().sync()


def close() -> None:
    """Shut down worker and close connections."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None
