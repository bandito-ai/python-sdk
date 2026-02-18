"""Shared helpers for TUI display."""

from __future__ import annotations

import json
from typing import Any


def format_response_text(
    value: Any,
    fallback: str = "[no response text]",
    *,
    max_length: int | None = None,
) -> str:
    """Format a response_text value for display.

    response_text is always stored as a dict after normalization.
    Simple ``{"response": "..."}`` wrappers are unwrapped to the plain
    string.  Richer dicts are pretty-printed as JSON.  Legacy string
    values are returned as-is.

    Args:
        value: The raw response_text value (dict, str, or None).
        fallback: Text to return when value is None.
        max_length: If set, truncate the result with "..." suffix.

    Returns:
        Display-ready string.
    """
    if value is None:
        display = fallback
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
