"""Help overlay — keybinding reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


HELP_TEXT = """\
[bold]Bandito TUI — Keyboard Shortcuts[/bold]

[bold]Navigation[/bold]
  [bold cyan]q[/]         Quit
  [bold cyan]Escape[/]    Go back
  [bold cyan]Enter[/]     Select / open detail
  [bold cyan]j / Down[/]  Move down
  [bold cyan]k / Up[/]    Move up
  [bold cyan]r[/]         Refresh data

[bold]Grading[/bold]
  [bold green]y[/]         Grade good (1.0)
  [bold red]n[/]         Grade bad (0.0)
  [bold cyan]s[/]         Skip event
  [bold cyan]Space[/]     Toggle select
  [bold cyan]a[/]         Select all
  [bold cyan]g[/]         Toggle graded/ungraded

[bold]View[/bold]
  [bold cyan]t[/]         Toggle sidebar
  [bold cyan]c[/]         Copy all (query + response + prompt)
  [bold cyan]?[/]         Toggle this help

[dim]Click any text section to copy it individually[/]
"""


class HelpScreen(ModalScreen):
    """Modal overlay showing keybinding help."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-dialog {
        width: 52;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="help-dialog"):
                yield Static(HELP_TEXT)
