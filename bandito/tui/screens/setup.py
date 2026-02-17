"""First-run setup — API key input and validation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static
from textual import work
from textual.worker import Worker, WorkerState


class SetupScreen(Screen):
    """Prompts user for API key on first run."""

    CSS = """
    SetupScreen {
        align: center middle;
    }

    #setup-container {
        width: 64;
        height: auto;
        padding: 2 3;
        border: thick $accent;
        background: $surface;
    }

    #setup-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #setup-subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }

    #api-key-input {
        margin-bottom: 1;
    }

    #base-url-input {
        margin-bottom: 1;
    }

    #setup-error {
        color: $error;
        margin-bottom: 1;
        display: none;
    }

    #setup-submit {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="setup-container"):
                yield Static("Bandito Setup", id="setup-title")
                yield Static(
                    "Enter your API key to get started.\n"
                    "Get one at https://bandito.dev",
                    id="setup-subtitle",
                )
                yield Label("API Key")
                yield Input(
                    placeholder="bnd_...",
                    password=True,
                    id="api-key-input",
                )
                yield Label("Base URL (optional)")
                yield Input(
                    placeholder="http://localhost:8000",
                    id="base-url-input",
                )
                yield Static("", id="setup-error")
                yield Button("Connect", variant="primary", id="setup-submit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-submit":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        api_key = self.query_one("#api-key-input", Input).value.strip()
        if not api_key:
            self._show_error("API key is required.")
            return

        base_url_input = self.query_one("#base-url-input", Input).value.strip()

        from bandito.config import DEFAULT_BASE_URL
        base_url = base_url_input or DEFAULT_BASE_URL

        self.query_one("#setup-submit", Button).disabled = True
        self.query_one("#setup-submit", Button).label = "Connecting..."
        self._hide_error()
        self._validate(api_key, base_url)

    @work(thread=True, exit_on_error=False)
    def _validate(self, api_key: str, base_url: str) -> tuple[str, str]:
        from bandito.tui.api import TuiAPI

        client = TuiAPI(api_key=api_key, base_url=base_url)
        try:
            client.list_bandits()
        finally:
            client.close()
        return api_key, base_url

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "_validate":
            return

        if event.state == WorkerState.SUCCESS:
            api_key, base_url = event.worker.result
            from bandito.config import save_config
            save_config(api_key, base_url)

            self.app._init_api(api_key, base_url)

            from bandito.tui.screens.bandit_select import BanditSelectScreen
            self.app.switch_screen(BanditSelectScreen())

        elif event.state == WorkerState.ERROR:
            error = event.worker.error
            base_url = self.query_one("#base-url-input", Input).value.strip()
            if not base_url:
                from bandito.config import DEFAULT_BASE_URL
                base_url = DEFAULT_BASE_URL
            if isinstance(error, Exception):
                import httpx
                if isinstance(error, httpx.HTTPStatusError):
                    status = error.response.status_code
                    if status == 401:
                        msg = "Invalid API key."
                    elif status == 403:
                        msg = "API key does not have access."
                    else:
                        msg = f"Server returned {status}."
                elif isinstance(error, httpx.ConnectError):
                    msg = f"Cannot connect to {base_url} — is the server running?"
                elif isinstance(error, httpx.TimeoutException):
                    msg = f"Connection to {base_url} timed out."
                else:
                    msg = str(error)
            else:
                msg = "Unknown error."
            self._show_error(msg)
            btn = self.query_one("#setup-submit", Button)
            btn.disabled = False
            btn.label = "Connect"

    def _show_error(self, text: str) -> None:
        error = self.query_one("#setup-error", Static)
        error.update(text)
        error.styles.display = "block"

    def _hide_error(self) -> None:
        error = self.query_one("#setup-error", Static)
        error.styles.display = "none"

    def action_quit(self) -> None:
        self.app.exit()
