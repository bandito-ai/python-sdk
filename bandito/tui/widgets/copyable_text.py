"""Clickable text widget that copies content to clipboard."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class CopyableText(Static):
    """A Static widget that copies its plain text content to clipboard on click."""

    CSS = """
    CopyableText {
        padding: 0 1;
    }

    CopyableText:hover {
        background: $accent 20%;
    }
    """

    def __init__(self, text: str, **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self._raw_text = text

    def on_click(self) -> None:
        try:
            self.app.copy_to_clipboard(self._raw_text)
            self.app.notify("Copied to clipboard", severity="information")
        except Exception:
            self.app.notify("Copy to clipboard failed", severity="warning")
