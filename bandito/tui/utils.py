"""Shared helpers for TUI display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_response(
    value: Any,
    fallback: str = "no response text",
    *,
    max_length: int | None = None,
) -> str:
    """Format a response value for display.

    Response is always stored as a dict after normalization.
    Simple ``{"response": "..."}`` wrappers are unwrapped to the plain
    string.  Richer dicts are pretty-printed as JSON.  Legacy string
    values are returned as-is.

    Args:
        value: The raw response value (dict, str, or None).
        fallback: Text to return when value is None.
        max_length: If set, truncate the result with "..." suffix.

    Returns:
        Display-ready string (Markdown-compatible).
    """
    if value is None:
        display = f"*{fallback}*" if fallback else ""
    elif isinstance(value, dict):
        if list(value.keys()) == ["response"] and isinstance(value["response"], str):
            display = value["response"]
        else:
            display = json.dumps(value, indent=2)
    else:
        display = str(value)

    if max_length is not None and len(display) > max_length:
        display = display[:max_length] + "..."

    return display


def save_event_json(event_data: dict[str, Any]) -> str:
    """Save event data as a JSON file in ~/.bandito/exports/.

    Returns the path to the saved file.
    """
    export_dir = Path.home() / ".bandito" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    uuid = event_data.get("local_event_uuid", "unknown")
    filename = f"event-{uuid[:12]}.json"
    path = export_dir / filename

    # Build a clean output dict with the fields that matter
    output: dict[str, Any] = {}
    keys = [
        "local_event_uuid", "bandit_id", "arm_id",
        "model_name", "model_provider", "system_prompt",
        "query_text", "response",
        "early_reward", "grade", "reward",
        "cost", "latency", "input_tokens", "output_tokens",
        "context_features", "created_at",
    ]
    for k in keys:
        if k in event_data:
            output[k] = event_data[k]

    path.write_text(json.dumps(output, indent=2, default=str))
    return str(path)
