"""Event detail — full display of query + response for grading."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from bandito.tui.utils import format_response_text
from bandito.tui.widgets.copyable_text import CopyableText


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
        Binding("c", "copy_all", "Copy All", show=True),
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
        imm_reward = ev.get("immediate_reward")
        reward_str = f"{imm_reward:.3f}" if imm_reward is not None else "—"

        query_text = str(ev.get("query_text") or "[no query text]")
        response_display = format_response_text(ev.get("response_text"))
        prompt = ev.get("system_prompt")

        with Vertical(id="detail-dialog"):
            yield Static(
                f"[bold]{model}[/] / {provider}  "
                f"[dim]cost:[/] {cost_str}  "
                f"[dim]latency:[/] {lat_str}  "
                f"[dim]reward:[/] {reward_str}  "
                f"[dim]{uuid[:12]}...[/]",
                id="detail-meta",
            )

            with VerticalScroll(id="detail-content"):
                yield Static("QUERY [dim italic](click to copy)[/]", classes="detail-section-header")
                yield CopyableText(query_text, classes="detail-text")
                yield Static("RESPONSE [dim italic](click to copy)[/]", classes="detail-section-header")
                yield CopyableText(response_display, classes="detail-text")

                if prompt:
                    yield Static("SYSTEM PROMPT [dim italic](click to copy)[/]", classes="detail-section-header")
                    yield CopyableText(str(prompt), classes="detail-text")

            yield Static(
                "[bold green]y[/] Good  [bold red]n[/] Bad  "
                "[bold]c[/] Copy all  [dim]Esc[/] Close  "
                "[dim]Click section to copy[/]",
                id="detail-footer",
            )

    def _build_full_text(self) -> str:
        """Build combined text of all sections for copy-all."""
        ev = self._event
        parts = []

        query = ev.get("query_text")
        if query:
            parts.append(f"QUERY:\n{query}")

        response = ev.get("response_text")
        if response:
            parts.append(f"RESPONSE:\n{format_response_text(response)}")

        prompt = ev.get("system_prompt")
        if prompt:
            parts.append(f"SYSTEM PROMPT:\n{prompt}")

        return "\n\n".join(parts)

    def action_copy_all(self) -> None:
        try:
            self.app.copy_to_clipboard(self._build_full_text())
            self.app.notify("Copied all to clipboard", severity="information")
        except Exception:
            self.app.notify("Copy to clipboard failed", severity="warning")

    def action_grade_good(self) -> None:
        self.dismiss(1.0)

    def action_grade_bad(self) -> None:
        self.dismiss(0.0)

    def action_dismiss_no_grade(self) -> None:
        self.dismiss(None)
