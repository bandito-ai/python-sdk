"""Event card — compact preview widget for grading queue list pane."""

from __future__ import annotations

import time
from typing import Any

from textual.app import ComposeResult
from textual.widgets import ListItem, Static


def _relative_time(epoch: float | None) -> str:
    """Return a human-friendly relative time string (e.g. '5m', '2h', '3d')."""
    if epoch is None:
        return ""
    delta = time.time() - epoch
    if delta < 0:
        return ""
    if delta < 60:
        return f"{int(delta)}s"
    if delta < 3600:
        return f"{int(delta / 60)}m"
    if delta < 86400:
        return f"{int(delta / 3600)}h"
    return f"{int(delta / 86400)}d"


class EventCard(ListItem):
    """Compact event preview for the narrow list pane.

    Renders as:
        [ ] gpt-4o/openai
          What is a contextual...

    Attributes:
        event_data: Raw event dict from local SQLite.
    """

    DEFAULT_CSS = """
    EventCard {
        height: auto;
        padding: 1 1;
    }

    EventCard .ec-model {
        color: $text;
        text-style: dim;
    }

    EventCard .ec-meta {
        color: $text-disabled;
    }

    EventCard.--skipped .ec-model {
        color: $text-disabled;
    }

    EventCard.--skipped .ec-meta {
        color: $text-disabled;
    }

    EventCard.--graded .ec-model {
        color: $text-muted;
    }

    EventCard.--graded .ec-meta {
        color: $text-disabled;
    }
    """

    def __init__(self, event_data: dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)
        self.event_data = event_data
        self._selected: bool = False
        self._skipped: bool = False
        self._graded: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", classes="ec-model")
        yield Static("", classes="ec-meta")

    def on_mount(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        model = self.event_data.get("model_name", "?")

        if self._graded:
            prefix = "[dim]\u2713[/]"
        elif self._skipped:
            prefix = "[S]"
        elif self._selected:
            prefix = "[x]"
        else:
            prefix = "[ ]"

        indicator = '' # " ►" if self.highlighted else "  "
        label = f"{model}"

        self.query_one(".ec-model", Static).update(
            f"{prefix}{indicator} [bold]{label}[/]"
        )

        # Meta line: relative time + immediate reward
        parts: list[str] = []
        created = self.event_data.get("created_at")
        if isinstance(created, str):
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = dt.timestamp()
            except (ValueError, TypeError):
                created = None
        rel = _relative_time(created)
        if rel:
            parts.append(rel)
        early = self.event_data.get("early_reward")
        if early is not None:
            parts.append(f"early reward: {early:.2f}")
        self.query_one(".ec-meta", Static).update(f"  {' · '.join(parts)}" if parts else "")

    @property
    def selected(self) -> bool:
        return self._selected

    def set_selected(self, value: bool) -> None:
        self._selected = value
        self._refresh_display()

    @property
    def skipped(self) -> bool:
        return self._skipped

    def set_skipped(self, value: bool) -> None:
        self._skipped = value
        if value:
            self.add_class("--skipped")
        else:
            self.remove_class("--skipped")
        self._refresh_display()

    @property
    def graded(self) -> bool:
        return self._graded

    def set_graded(self, value: bool) -> None:
        self._graded = value
        if value:
            self.add_class("--graded")
        else:
            self.remove_class("--graded")
        self._refresh_display()
