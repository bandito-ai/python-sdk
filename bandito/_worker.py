"""Payload utilities for cloud event ingestion."""

from __future__ import annotations

_TEXT_FIELDS = ("query_text", "response")
_METADATA_FIELDS = ("model_name", "model_provider")


def prepare_cloud_payload(events: list[dict], *, include_text: bool) -> list[dict]:
    """Return shallow copies of events ready for cloud ingest.

    Always strips model_name/model_provider (only needed in local SQLite for TUI).
    Strips query_text/response when include_text is False (data_storage="local").
    """
    stripped = []
    for e in events:
        copy = e.copy()
        for field in _METADATA_FIELDS:
            copy.pop(field, None)
        if not include_text:
            for field in _TEXT_FIELDS:
                copy.pop(field, None)
        stripped.append(copy)
    return stripped
