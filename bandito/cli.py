"""CLI entry point — routes subcommands with lazy imports."""

import sys

USAGE = """\
Usage: bandito <command>

Commands:
  init     Configure API key and validate connection
  create   Create a new bandit with arms
  ui       Launch the TUI scoring workbench
  help     Show this help message
"""


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "init":
        from bandito.cli_init import run_init

        run_init()

    elif command == "create":
        from bandito.cli_create import run_create

        run_create()

    elif command == "ui":
        from bandito.tui.app import BanditoApp

        app = BanditoApp()
        app.run()

    elif command in ("help", "--help", "-h"):
        print(USAGE)

    else:
        print(f"Unknown command: {command}\n")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
