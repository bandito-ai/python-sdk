"""Bandit selector — landing screen after auth."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static
from textual import work
from textual.worker import Worker, WorkerState


class BanditSelectScreen(Screen):
    """Lists all bandits; user picks one to open the dashboard."""

    CSS = """
    BanditSelectScreen {
        layout: vertical;
    }

    #bandit-title {
        text-style: bold;
        padding: 1 2;
    }

    #bandit-status {
        color: $text-muted;
        padding: 0 2 1 2;
    }

    #bandit-table {
        height: 1fr;
        margin: 0 2;
    }

    #bandit-empty {
        padding: 2;
        text-align: center;
        color: $text-muted;
        display: none;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("question_mark", "show_help", "Help", show=True, key_display="?"),
        Binding("q", "quit", "Quit", show=True),
        Binding("escape", "quit", "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Select a Bandit", id="bandit-title")
            yield Static("Loading...", id="bandit-status")
            yield DataTable(id="bandit-table", cursor_type="row")
            yield Static("No bandits found. Create one via the API first.", id="bandit-empty")

    def on_mount(self) -> None:
        table = self.query_one("#bandit-table", DataTable)
        table.add_columns("Name", "Type", "Arms", "Pulls", "Mode")
        self._load_bandits()

    @work(thread=True, exit_on_error=False)
    def _load_bandits(self) -> list[dict]:
        data = self.app.api.list_bandits()
        return data.get("items", [])

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "_load_bandits":
            return

        if event.state == WorkerState.SUCCESS:
            bandits = event.worker.result
            self._bandits = bandits
            table = self.query_one("#bandit-table", DataTable)
            table.clear()
            if not bandits:
                self.query_one("#bandit-empty").styles.display = "block"
                self.query_one("#bandit-status").update("")
            else:
                self.query_one("#bandit-empty").styles.display = "none"
                for b in bandits:
                    table.add_row(
                        b["name"],
                        b.get("type", "online"),
                        str(b.get("arm_count", 0)),
                        str(b.get("total_pull_count", 0)),
                        b.get("optimization_mode", "base"),
                        key=str(b["id"]),
                    )
                self.query_one("#bandit-status").update(
                    f"{len(bandits)} bandit{'s' if len(bandits) != 1 else ''}"
                )

        elif event.state == WorkerState.ERROR:
            error = event.worker.error
            import httpx
            if isinstance(error, httpx.ConnectError):
                msg = "Cannot connect to server."
            elif isinstance(error, httpx.HTTPStatusError):
                msg = f"Server returned {error.response.status_code}."
            else:
                msg = str(error)
            self.app.notify(msg, title="Failed to load bandits", severity="error")
            self.query_one("#bandit-status").update(f"Error: {msg}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not hasattr(self, "_bandits"):
            return
        bandit_id = int(str(event.row_key.value))
        bandit = next((b for b in self._bandits if b["id"] == bandit_id), None)
        if bandit:
            from bandito.tui.screens.dashboard import DashboardScreen
            self.app.push_screen(DashboardScreen(bandit))

    def action_refresh(self) -> None:
        self.query_one("#bandit-status").update("Loading...")
        self._load_bandits()

    def action_show_help(self) -> None:
        from bandito.tui.screens.help import HelpScreen
        self.app.push_screen(HelpScreen())

    def action_quit(self) -> None:
        self.app.exit()
