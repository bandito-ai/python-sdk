"""bandito init — configure API key and validate connection."""

from __future__ import annotations

import getpass
import os
import sys

from bandito.config import CONFIG_FILE, DEFAULT_BASE_URL, save_config


def run_init() -> None:
    # 1. Warn if overwriting existing config
    if CONFIG_FILE.exists():
        print(f"  Existing config found at {CONFIG_FILE}")
        answer = input("  Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("  Aborted.")
            return

    # 2. Read API key from env or prompt
    api_key = os.environ.get("BANDITO_API_KEY", "").strip()
    if api_key:
        print("  Using API key from BANDITO_API_KEY env var.")
    else:
        api_key = getpass.getpass("  Enter your API key: ").strip()

    if not api_key:
        print("  Error: API key is required.")
        sys.exit(1)

    # 3. Prompt for base URL
    base_url_input = input(
        f"  Base URL [{DEFAULT_BASE_URL}]: "
    ).strip()
    base_url = base_url_input or DEFAULT_BASE_URL

    # 4. Validate connection
    print(f"  Connecting to {base_url}...")
    try:
        from bandito.http import BanditoHTTP

        http = BanditoHTTP(base_url=base_url, api_key=api_key)
        try:
            http.list_bandits()
        finally:
            http.close()
    except Exception as exc:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                print("  Error: Invalid API key.")
            else:
                print(f"  Error: Server returned {status}.")
        elif isinstance(exc, httpx.ConnectError):
            print(f"  Error: Cannot connect to {base_url} — is the server running?")
        elif isinstance(exc, httpx.TimeoutException):
            print(f"  Error: Connection to {base_url} timed out.")
        else:
            print(f"  Error: {exc}")
        sys.exit(1)

    print("  Connected!")

    # 5. Save config
    save_config(api_key, base_url)
    print(f"  Config saved to {CONFIG_FILE}")

    # 6. Next steps
    print()
    print("  Next steps:")
    print("    1. Create a bandit:  bandito create")
    print("    2. Use the SDK:")
    print()
    print("       import bandito")
    print('       bandito.connect(api_key="...")')
    print('       result = bandito.pull("my-bandit", query="hello")')
    print("       # call result.model / result.prompt")
