"""Event detail — full display of query + response for grading."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from bandito.tui.utils import format_response


class EventDetailScreen(ModalScreen[None]):
    """Modal showing full event payload. Returns grade or None."""

    CSS = """
    EventDetailScreen {
        align: center middle;
    }

    #detail-dialog {
        width: 90%;
        height: 85%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }

    #detail-meta {
        height: auto;
        padding: 0 0 1 0;
        border-bottom: solid $accent;
        margin-bottom: 1;
    }

    #detail-content {
        height: 1fr;
    }

    .detail-section-header {
        text-style: bold;
        margin-top: 1;
        color: $accent;
    }

    .detail-text {
        margin: 0 0 1 0;
    }

    .detail-markdown {
        margin: 0 0 1 0;
    }

    #detail-footer {
        height: auto;
        padding: 1 0 0 0;
        border-top: solid $accent;
        margin-top: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_no_grade", "Close", show=True),
        Binding("y", "grade_good", "Good", show=True),
        Binding("n", "grade_bad", "Bad", show=True),
        Binding("d", "download_event", "Download", show=True),
        Binding("left", "prev_event", "Prev", show=True),
        Binding("right", "next_event", "Next", show=True),
    ]

    def __init__(
        self,
        event_data: dict[str, Any],
        events: list[dict[str, Any]] | None = None,
        index: int = 0,
    ) -> None:
        super().__init__()
        self._events = events or [event_data]
        self._index = index
        self._event = self._events[self._index]

    def compose(self) -> ComposeResult:
        ev = self._event
        uuid = ev.get("local_event_uuid", "?")
        model = ev.get("model_name", "?")
        provider = ev.get("model_provider", "?")

        cost = ev.get("cost")
        cost_str = f"${cost:.4f}" if cost is not None else "—"
        latency = ev.get("latency")
        lat_str = f"{latency:.2f}s" if latency is not None else "—"
        early = ev.get("early_reward")
        reward_str = f"{early:.3f}" if early is not None else "—"

        query_text = str(ev.get("query_text") or "*no query text*")
        response_display = format_response(ev.get("response"))
        prompt = ev.get("system_prompt")

        with Vertical(id="detail-dialog"):
            yield Static(
                f"[bold]{model}[/] / {provider}  "
                f"[dim]cost:[/] {cost_str}  "
                f"[dim]latency:[/] {lat_str}  "
                f"[dim]reward:[/] {reward_str}",
                id="detail-meta",
            )

            with VerticalScroll(id="detail-content"):
                yield Static(f"[dim]event_id:[/] {uuid}", id="detail-event-uuid", classes="detail-text")
                yield Static("USER INPUT", classes="detail-section-header")
                yield Markdown(query_text, id="detail-query-text", classes="detail-markdown")
                yield Static("RESPONSE", classes="detail-section-header")
                yield Markdown(response_display, id="detail-response-text", classes="detail-markdown")

                yield Static("SYSTEM PROMPT", id="detail-prompt-header", classes="detail-section-header")
                yield Markdown(str(prompt) if prompt else "", id="detail-prompt-text", classes="detail-markdown")

            yield Static(
                "[bold green]y[/] Good  [bold red]n[/] Bad  "
                "[bold]d[/] Download  [dim]←/→[/] Navigate  [dim]Esc[/] Close",
                id="detail-footer",
            )

    def on_mount(self) -> None:
        prompt = self._event.get("system_prompt")
        if not prompt:
            self.query_one("#detail-prompt-header", Static).styles.display = "none"
            self.query_one("#detail-prompt-text", Markdown).styles.display = "none"

    def _refresh_display(self) -> None:
        """Update all displayed content to match current event."""
        ev = self._event
        model = ev.get("model_name", "?")
        provider = ev.get("model_provider", "?")

        cost = ev.get("cost")
        cost_str = f"${cost:.4f}" if cost is not None else "—"
        latency = ev.get("latency")
        lat_str = f"{latency:.2f}s" if latency is not None else "—"
        early = ev.get("early_reward")
        reward_str = f"{early:.3f}" if early is not None else "—"

        self.query_one("#detail-meta", Static).update(
            f"[bold]{model}[/] / {provider}  "
            f"[dim]cost:[/] {cost_str}  "
            f"[dim]latency:[/] {lat_str}  "
            f"[dim]reward:[/] {reward_str}",
        )

        query_text = str(ev.get("query_text") or "*no query text*")
        response_display = format_response(ev.get("response"))
        prompt = ev.get("system_prompt")

        self.query_one("#detail-query-text", Markdown).update(query_text)
        self.query_one("#detail-response-text", Markdown).update(response_display)

        prompt_header = self.query_one("#detail-prompt-header", Static)
        prompt_md = self.query_one("#detail-prompt-text", Markdown)
        if prompt:
            prompt_header.styles.display = "block"
            prompt_md.styles.display = "block"
            prompt_md.update(str(prompt))
        else:
            prompt_header.styles.display = "none"
            prompt_md.styles.display = "none"

        self.query_one("#detail-event-uuid", Static).update(
            f"[dim]event_id:[/] {ev.get('local_event_uuid', '?')}"
        )

        self.query_one("#detail-content", VerticalScroll).scroll_home(animate=False)

    def action_prev_event(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._event = self._events[self._index]
            self._refresh_display()

    def action_next_event(self) -> None:
        if self._index < len(self._events) - 1:
            self._index += 1
            self._event = self._events[self._index]
            self._refresh_display()

    def action_download_event(self) -> None:
        from bandito.tui.utils import save_event_json
        path = save_event_json(self._event)
        self.app.notify(f"Saved to {path}", severity="information")

    class Graded(Message):
        """Posted when an event is graded inside the detail modal."""
        def __init__(self, uuid: str, grade: float) -> None:
            super().__init__()
            self.uuid = uuid
            self.grade = grade

    def _grade_current(self, reward: float) -> None:
        uuid = self._event.get("local_event_uuid", "")
        if not uuid:
            return
        self.post_message(self.Graded(uuid, reward))
        # Remove graded event from local list
        self._events = [e for e in self._events if e.get("local_event_uuid") != uuid]
        if not self._events:
            self.dismiss(None)
            return
        # Clamp index (if we were at the end, step back)
        if self._index >= len(self._events):
            self._index = len(self._events) - 1
        self._event = self._events[self._index]
        self._refresh_display()

    def action_grade_good(self) -> None:
        self._grade_current(1.0)

    def action_grade_bad(self) -> None:
        self._grade_current(0.0)

    def action_dismiss_no_grade(self) -> None:
        self.dismiss(None)
