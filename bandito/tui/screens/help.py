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
  [bold cyan]Tab[/]       Next tab
  [bold cyan]Shift+Tab[/] Previous tab

[bold]Grading[/bold]
  [bold green]y[/]         Grade good (1.0)
  [bold red]n[/]         Grade bad (0.0)

[bold]Lists[/bold]
  [bold cyan]j / Down[/]  Move down
  [bold cyan]k / Up[/]    Move up
  [bold cyan]r[/]         Refresh data

[bold cyan]?[/]         Toggle this help
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
