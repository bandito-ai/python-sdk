"""Bandito TUI — terminal scoring workbench."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from bandito.store import EventStore
from bandito.tui.api import TuiAPI
from bandito.config import load_config

# Default store path — shared with the SDK
_DEFAULT_STORE_PATH = str(__import__("pathlib").Path.home() / ".bandito" / "events.db")


class BanditoApp(App):
    """Main Bandito TUI application."""

    TITLE = "Bandito"
    SUB_TITLE = "LLM Optimization Workbench"

    CSS = """
    Screen {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "help", "Help", show=True, key_display="?"),
    ]

    api: TuiAPI | None = None
    store: EventStore | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        config = load_config()
        if not config.api_key:
            from bandito.tui.screens.setup import SetupScreen
            self.push_screen(SetupScreen())
        else:
            self._init_api(config.api_key, config.base_url)
            from bandito.tui.screens.bandit_select import BanditSelectScreen
            self.push_screen(BanditSelectScreen())

    def _init_api(self, api_key: str, base_url: str) -> None:
        if self.api is not None:
            self.api.close()
        self.api = TuiAPI(api_key=api_key, base_url=base_url)
        self._init_store()

    def _init_store(self) -> None:
        """Open the shared local SQLite store (read-only for grading)."""
        if self.store is not None:
            self.store.close()
        import os
        if os.path.exists(_DEFAULT_STORE_PATH):
            self.store = EventStore(_DEFAULT_STORE_PATH)
        else:
            self.store = None

    def action_help(self) -> None:
        from bandito.tui.screens.help import HelpScreen
        self.push_screen(HelpScreen())

    def on_unmount(self) -> None:
        if self.api is not None:
            self.api.close()
        if self.store is not None:
            self.store.close()
