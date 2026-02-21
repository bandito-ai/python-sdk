"""Bandito SDK — contextual bandit optimization for LLM selection.

Recommended (context manager):
    from bandito import BanditoClient

    with BanditoClient(api_key="bnd_...") as client:
        result = client.pull("my-chatbot", query=user_message)
        response = call_llm(result.model, result.prompt, user_message)
        client.update(
            result,
            response=response.text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

Module-level singleton (convenience):
    import bandito
    bandito.connect()
    result = bandito.pull("my-chatbot", query=user_message)
    ...
    bandito.close()
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
    "grade",
    "sync",
    "close",
]

_client: BanditoClient | None = None
_lock = threading.Lock()


def _get_client() -> BanditoClient:
    with _lock:
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


def update(
    pull_result: PullResult,
    *,
    query_text: str | None = None,
    response: str | dict | None = None,
    reward: float | None = None,
    cost: float | None = None,
    latency: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    segment: dict[str, str] | None = None,
    failed: bool = False,
) -> None:
    """Send event data to cloud (writes to SQLite first)."""
    _get_client().update(
        pull_result,
        query_text=query_text,
        response=response,
        reward=reward,
        cost=cost,
        latency=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        segment=segment,
        failed=failed,
    )


def grade(event_id: str, grade: float) -> None:
    """Send a human grade for an existing event."""
    _get_client().grade(event_id, grade)


def sync() -> None:
    """Explicit state refresh from cloud."""
    _get_client().sync()


def close() -> None:
    """Shut down executor and close connections."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None
