"""bandito create — open the web dashboard to create a bandit."""

from __future__ import annotations

import webbrowser

DASHBOARD_URL = "https://usebandito.com/bandits"


def run_create() -> None:
    print(f"  Opening {DASHBOARD_URL} to create a bandit...")
    webbrowser.open(DASHBOARD_URL)
