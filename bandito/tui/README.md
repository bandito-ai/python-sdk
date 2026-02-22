# Bandito TUI

Terminal scoring workbench for grading LLM responses. Ships inside the SDK package.

## Quick Start

```bash
# From sdk/
uv sync
uv pip install -e .
bandito tui
```

For local development, set the base URL in `~/.bandito/config.toml`:

```toml
api_key = "bnd_your_key_here"
base_url = "http://localhost:8000"
```

Or via environment variables:

```bash
export BANDITO_API_KEY="bnd_..."
export BANDITO_BASE_URL="http://localhost:8000"
```

## Screen Flow

```
SetupScreen (first-run only, if no API key)
    |
BanditSelectScreen (landing page)
    | Enter
DashboardScreen (split-pane: event list + detail, toggleable sidebar)
    |                              | t
    | Enter                    Stats/Arms sidebar
EventDetailScreen (modal)
```

## Keyboard Shortcuts

| Key             | Action                          |
|-----------------|---------------------------------|
| `y`             | Grade good (1.0)                |
| `n`             | Grade bad (0.0)                 |
| `s`             | Skip (move to end of list)      |
| `Space`         | Toggle selection (batch grading) |
| `a`             | Select all                      |
| `r`             | Refresh data                    |
| `t`             | Toggle stats/arms sidebar       |
| `g`             | Toggle showing graded events    |
| `d`             | Download event as JSON          |
| `j` / `k`       | Navigate lists                  |
| `Enter`         | Select / open detail            |
| `Esc`           | Go back                         |
| `?`             | Help overlay                    |
| `q`             | Quit                            |

## Screens

### Setup Screen
First-run only. Prompts for API key and optional base URL. Validates by calling the bandits list endpoint. Saves config to `~/.bandito/config.toml`.

### Bandit Selector
Lists all bandits with name, type, arm count, pull count, and optimization mode. Select one to open the dashboard.

### Dashboard
Split-pane grading interface with event list and detail view. Press `t` to toggle the stats/arms sidebar.

**Event list** — Ungraded events from local SQLite. Each card shows model, provider, cost, latency, and a truncated query. Press `y`/`n` to grade inline, `Space` to select for batch grading, or `Enter` to open the full event detail.

**Sidebar** (toggle with `t`) — Stats summary cards (total events, average reward, total cost, budget) and per-arm performance table sorted by pull share.

### Event Detail
Modal showing the full query text, response text, and system prompt. Grade with `y`/`n` or dismiss with `Esc`.

## Data Flow

**Grading requires cloud connectivity.** The local SQLite provides event content for display, but the grade must reach the cloud (where the Bayesian state lives) to update arm selection.

```
SDK writes events ──→ Local SQLite (~/.bandito/events.db)
                              │
TUI reads event content ──────┘  (query_text, response, model_name)
    │
    ├── Display: fully local, no cloud round-trip for response text
    │
    └── Grade submission:
        ├── 1. Mark graded locally (human_reward + graded_at in SQLite)
        └── 2. PATCH /events/{uuid}/grade to cloud API
                └── Cloud runs Bayesian update, distributes new weights
```

| Feature | Reads from | Writes to |
|---------|------------|-----------|
| Grading display | Local SQLite | — |
| Grade submit | — | Local SQLite + Cloud API |
| Stats | Cloud API | — |
| Arm performance | Cloud API | — |

Press `r` to refresh the grading queue and pick up new events from the SDK.

## Architecture

```
sdk/bandito/
    cli.py                    # `bandito` CLI dispatcher
    tui/
        app.py                # Textual App, CSS, screen routing
        config.py             # TOML config loader (~/.bandito/config.toml)
        api.py                # TUI HTTP client (analytics, bandits, reward)
        screens/
            setup.py          # First-run API key input
            bandit_select.py  # Bandit list / picker
            dashboard.py      # Split-pane grading workbench
            event_detail.py   # Full event detail modal
            help.py           # Keybinding reference overlay
        widgets/
            stats_panel.py    # Stats summary cards
            arm_table.py      # Arm performance DataTable
            event_card.py     # Event preview for grading queue
```

The TUI uses Textual's `@work(thread=True)` to bridge sync httpx calls into the async Textual event loop. All API calls happen in background threads so the UI stays responsive.
