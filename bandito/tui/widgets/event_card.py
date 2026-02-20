"""Event card — compact preview widget for grading queue list pane."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import ListItem, Static


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
        padding: 0 1;
    }

    EventCard .ec-model {
        color: $text;
    }

    EventCard .ec-query {
        color: $text-muted;
    }

    EventCard.--skipped .ec-model {
        color: $text-disabled;
    }

    EventCard.--skipped .ec-query {
        color: $text-disabled;
    }

    EventCard.--graded .ec-model {
        color: $text-muted;
    }

    EventCard.--graded .ec-query {
        color: $text-disabled;
    }
    """

    MAX_QUERY_LEN = 23

    def __init__(self, event_data: dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)
        self.event_data = event_data
        self._selected: bool = False
        self._skipped: bool = False
        self._graded: bool = False

    def compose(self) -> ComposeResult:
        yield Static("", classes="ec-model")
        yield Static("", classes="ec-query")

    def on_mount(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        model = self.event_data.get("model_name", "?")
        provider = self.event_data.get("model_provider", "")

        if self._graded:
            prefix = "[dim]\u2713[/]"
        elif self._skipped:
            prefix = "[S]"
        elif self._selected:
            prefix = "[x]"
        else:
            prefix = "[ ]"

        indicator = " ►" if self.highlighted else "  "
        label = f"{model}/{provider}" if provider else model

        imm_reward = self.event_data.get("immediate_reward")
        reward_tag = f" [dim]r:{imm_reward:.2f}[/]" if imm_reward is not None else ""

        self.query_one(".ec-model", Static).update(
            f"{prefix}{indicator} [bold]{label}[/]{reward_tag}"
        )

        query = self.event_data.get("query_text", "")
        truncated = query[:self.MAX_QUERY_LEN]
        if len(query) > self.MAX_QUERY_LEN:
            truncated += "..."
        if self._graded:
            reward = self.event_data.get("human_reward")
            if reward is not None and reward >= 0.5:
                truncated = f"[dim]graded good[/]"
            elif reward is not None:
                truncated = f"[dim]graded bad[/]"
            else:
                truncated = "[dim]graded[/]"
        elif self._skipped:
            truncated = "(skipped)"

        self.query_one(".ec-query", Static).update(f"  {truncated}")

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
