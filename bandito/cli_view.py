"""bandito view <name> — open a bandit in the web dashboard by name."""

from __future__ import annotations

import sys
import webbrowser

from bandito.config import load_config
from bandito.http import BanditoHTTP

DASHBOARD_BASE = "https://usebandito.com/bandits"


def run_view(name: str | None) -> None:
    if not name:
        print("Usage: bandito view <bandit-name>")
        sys.exit(1)

    config = load_config()
    if not config.api_key:
        print("No API key configured. Run `bandito init` first.")
        sys.exit(1)

    http = BanditoHTTP(base_url=config.base_url, api_key=config.api_key)
    try:
        data = http.list_bandits()
    except Exception as exc:
        print(f"Failed to fetch bandits: {exc}")
        sys.exit(1)
    finally:
        http.close()

    bandits = data.get("bandits", [])
    match = next((b for b in bandits if b["name"] == name), None)

    if not match:
        available = [b["name"] for b in bandits]
        print(f'Bandit "{name}" not found.')
        if available:
            print(f"Available bandits: {', '.join(available)}")
        sys.exit(1)

    url = f"{DASHBOARD_BASE}/{match['id']}"
    print(f"  Opening {url} ...")
    webbrowser.open(url)
