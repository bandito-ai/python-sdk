"""Event detail — full display of query + response for grading."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from bandito.tui.utils import format_response


class EventDetailScreen(ModalScreen[float | None]):
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
    ]

    def __init__(self, event_data: dict[str, Any]) -> None:
        super().__init__()
        self._event = event_data

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
                yield Static(f"[dim]event_id:[/] {uuid}", classes="detail-text")
                yield Static("USER INPUT", classes="detail-section-header")
                yield Markdown(query_text, classes="detail-markdown")
                yield Static("RESPONSE", classes="detail-section-header")
                yield Markdown(response_display, classes="detail-markdown")

                if prompt:
                    yield Static("SYSTEM PROMPT", classes="detail-section-header")
                    yield Markdown(str(prompt), classes="detail-markdown")

            yield Static(
                "[bold green]y[/] Good  [bold red]n[/] Bad  "
                "[bold]d[/] Download  [dim]Esc[/] Close",
                id="detail-footer",
            )

    def action_download_event(self) -> None:
        from bandito.tui.utils import save_event_json
        path = save_event_json(self._event)
        self.app.notify(f"Saved to {path}", severity="information")

    def action_grade_good(self) -> None:
        self.dismiss(1.0)

    def action_grade_bad(self) -> None:
        self.dismiss(0.0)

    def action_dismiss_no_grade(self) -> None:
        self.dismiss(None)
