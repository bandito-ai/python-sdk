# Bandito Python SDK

Provider-agnostic contextual bandit optimization for LLM model and prompt selection. Zero-latency decisions via local Thompson Sampling, with crash-safe event durability and background cloud sync.

Includes a terminal UI for reviewing and grading LLM responses.

## Installation

```bash
uv add git+https://github.com/bandito-ai/python-sdk.git
```
or

```bash
pip install git+https://github.com/bandito-ai/python-sdk.git
```


Requires Python 3.12+.

## Getting Started

### 1. Configure your API key

```bash
bandito init
```

Walks you through setup interactively — prompts for your API key (or reads `BANDITO_API_KEY` from the environment), validates the connection, and saves config to `~/.bandito/config.toml`.

### 2. Create a bandit

```bash
bandito create
```

Opens the Bandito web dashboard where you can create bandits and configure arms (model + provider + system prompt combinations).

### 3. Optimize

```python
import bandito

bandito.connect(api_key="bnd_...")

result = bandito.pull("my-chatbot", query=user_message)

response = openai.chat.completions.create(
    model=result.model,
    messages=[
        {"role": "system", "content": result.prompt},
        {"role": "user", "content": user_message},
    ],
)

bandito.update(
    result,
    query_text=user_message,
    response=response.choices[0].message.content,
    reward=0.85,
    cost=0.003,
    latency=elapsed_ms,
)
```

### 4. Grade responses

```bash
bandito tui
```

Opens the terminal scoring workbench. Review LLM responses and grade them with `y` (good) / `n` (bad). Grades feed back into the bandit to improve future arm selection.

## CLI Reference

```
Usage: bandito <command>

Commands:
  init     Configure API key and validate connection
  create   Create a new bandit with arms
  tui      Launch the TUI scoring workbench
  help     Show this help message
```

## SDK API

### Two usage patterns

**Module-level singleton** (simplest):

```python
import bandito

bandito.connect(api_key="bnd_...")
result = bandito.pull("my-chatbot")
```

**Explicit client with context manager** (recommended):

```python
from bandito import BanditoClient

with BanditoClient(api_key="bnd_...") as client:
    result = client.pull("my-chatbot")
```

**Explicit connect/close** (testing, long-running servers):

```python
from bandito import BanditoClient

client = BanditoClient(api_key="bnd_...")
client.connect()
result = client.pull("my-chatbot")
client.close()
```

### `connect(api_key=None, **kwargs)`

Bootstrap the SDK. Authenticates with the cloud, fetches all bandit state, and flushes any pending events from a previous crash.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `BANDITO_API_KEY` env | API key for authentication |
| `base_url` | `str` | `http://localhost:8000` | Cloud API base URL |
| `store_path` | `str` | `~/.bandito/events.db` | Path to SQLite file for event durability |
| `data_storage` | `str` | `"local"` | `"local"` keeps query/response text local-only; `"cloud"` sends it to the server |

### `pull(bandit_name, *, query=None) -> PullResult`

Local Thompson Sampling decision. Pure math, <1ms, no network call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bandit_name` | `str` | Name of the bandit to pull from |
| `query` | `str` | User query text (used for feature engineering) |

Returns a `PullResult` with:
- `result.model` — model name (e.g. `"gpt-4o"`)
- `result.prompt` — system prompt text
- `result.event_id` — UUID linking this pull to its update/grade
- `result.arm` — full `Arm` object (model_name, model_provider, system_prompt, is_prompt_templated)
- `result.scores` — `dict[int, float]` of arm_id to score (for debugging)

### `update(pull_result, **kwargs)`

Report event data. Writes to local SQLite first (crash-safe), then submits a non-blocking background flush to cloud.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pull_result` | `PullResult` | Result from `pull()` |
| `query_text` | `str` | The user's query |
| `response` | `str \| dict` | The LLM's response. Strings are auto-wrapped as `{"response": "..."}` |
| `reward` | `float` | Immediate reward (0.0-1.0) |
| `cost` | `float` | Cost in dollars |
| `latency` | `float` | Latency in milliseconds |
| `input_tokens` | `int` | Input token count (auto-calculates cost if `cost` not provided) |
| `output_tokens` | `int` | Output token count |
| `segment` | `dict[str, str]` | Key-value segment tags |

### `grade(event_id, grade)`

Send a human grade for an existing event. Synchronous HTTP call — blocks until confirmed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `event_id` | `str` | The `event_id` from `PullResult` |
| `grade` | `float` | Grade value (0.0-1.0) |

### `sync()`

Explicit state refresh from cloud. Call periodically (e.g. every 30s) in long-running servers to pick up updated weights.

### `close()`

Shut down the background flush executor, flush remaining events, and close all connections.

## TUI Scoring Workbench

Launch with `bandito tui`. On first run it prompts for your API key if `~/.bandito/config.toml` doesn't exist yet.

**Screens:**
- **Bandit selector** — pick which bandit to review
- **Dashboard** — split-pane grading interface with event list, detail view, and toggleable stats sidebar

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `y` | Grade good (1.0) |
| `n` | Grade bad (0.0) |
| `s` | Skip (move to end of list) |
| `Space` | Toggle selection (for batch grading) |
| `a` | Select all |
| `r` | Refresh |
| `t` | Toggle stats/arms sidebar |
| `g` | Toggle showing graded events |
| `j`/`k` | Move cursor down/up |
| `Enter` | Open event detail |
| `Escape` | Go back |
| `?` | Help |
| `q` | Quit |

## How It Works

The SDK caches the Bayesian posterior (θ, Cholesky factor) for each bandit locally. On `pull()`, it:

1. Samples θ&#771; from the shared posterior via Thompson Sampling
2. Builds a feature vector per arm (model one-hot + prompt one-hot + query length interaction + latency interaction)
3. Scores each arm: score = x&#7511; · θ&#771;
4. Returns the highest-scoring arm

All Bayesian updates happen server-side. The SDK is a pure read cache that periodically refreshes via heartbeat.

**Crash safety** — Events are written to a local SQLite database (WAL mode) before any network call. If the process crashes, pending events are retried on the next `connect()`.

**Fail-safe** — If the cloud is unreachable after initial connect, the SDK continues making decisions with the last-known-good weights. Your application never breaks due to a Bandito outage.

## Configuration

Config is stored at `~/.bandito/config.toml` (created by `bandito init`):

```toml
api_key = "bnd_..."
base_url = "http://localhost:8000"
data_storage = "cloud"  # omitted when using the default ("local")
```

When `data_storage = "local"` (the default), query and response text are stored only in the local SQLite database and never sent to the cloud. Set to `"cloud"` to include text in server-side events.

Environment variables override the config file:

| Variable | Description |
|----------|-------------|
| `BANDITO_API_KEY` | API key for authentication |
| `BANDITO_BASE_URL` | Cloud API base URL |
| `BANDITO_DATA_STORAGE` | `"local"` (default) or `"cloud"` — controls whether query/response text is sent to the server |

## Architecture

```
bandito/
├── __init__.py        # Module-level API (lazy singleton)
├── client.py          # BanditoClient orchestrator
├── config.py          # Config loader (TOML + env vars)
├── models.py          # Arm, PullResult
├── http.py            # Sync httpx transport
├── store.py           # SQLite WAL event store
├── _worker.py         # Cloud payload utilities
├── cli.py             # CLI dispatcher (init, create, ui)
├── cli_init.py        # bandito init
├── cli_create.py      # bandito create
├── engine/            # Pure math (copied from backend)
│   ├── constants.py
│   ├── features.py
│   └── linalg.py
└── tui/               # Terminal UI (Textual)
    ├── app.py
    ├── api.py
    ├── screens/
    └── widgets/
```

## Development

```bash
uv sync              # Install dependencies
uv run pytest -q     # Run tests (95 tests)
```
