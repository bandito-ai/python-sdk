# Bandito Python SDK

Contextual bandit optimization for LLM model and prompt selection. Zero-latency decisions via local Thompson Sampling, with crash-safe event durability and background cloud sync.

## Installation

```bash
pip install bandito
```

## Quick Start

```python
import bandito

# Connect to cloud (reads BANDITO_API_KEY from env, or pass explicitly)
bandito.connect(api_key="bnd_...")

# Pull — local Thompson Sampling, <1ms, no network
result = bandito.pull("my-chatbot", query=user_message)

# Use the chosen arm
response = openai.chat.completions.create(
    model=result.model,
    messages=[
        {"role": "system", "content": result.prompt},
        {"role": "user", "content": user_message},
    ],
)

# Send event to cloud — writes to local SQLite first, then flushes async
bandito.update(
    result,
    query_text=user_message,
    response_text=response.choices[0].message.content,
    reward=0.85,
    cost=0.003,
    latency=elapsed_ms,
)

# Optional: delayed human reward
bandito.reward(result.event_id, reward=0.9)

# Optional: explicit state refresh (background thread handles this automatically)
bandito.sync()
```

## Usage Patterns

### Module-level singleton (simplest)

```python
import bandito

bandito.connect(api_key="bnd_...")
result = bandito.pull("my-chatbot")
```

### Explicit client (testing, multiple instances, DI)

```python
from bandito import BanditoClient

client = BanditoClient(api_key="bnd_...")
client.connect()
result = client.pull("my-chatbot")
```

## API Reference

### `bandito.connect(api_key=None, **kwargs)`

Bootstrap the SDK. Authenticates with the cloud, fetches all bandit state, flushes any pending events from a previous crash, and starts the background sync worker.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `BANDITO_API_KEY` env | API key for authentication |
| `sync_interval` | `float` | `30.0` | Seconds between background heartbeats |
| `flush_interval` | `float` | `5.0` | Seconds between background event flushes |
| `store_path` | `str` | `~/.bandito/events.db` | Path to SQLite file for event durability |

### `bandito.pull(bandit_name, *, query=None) -> PullResult`

Local Thompson Sampling decision. Pure math, <1ms, no network call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bandit_name` | `str` | Name of the bandit to pull from |
| `query` | `str` | User query text (used for feature engineering) |

Returns a `PullResult` with:
- `result.arm` — the chosen `Arm` object
- `result.model` — convenience for `arm.model_name`
- `result.prompt` — convenience for `arm.system_prompt`
- `result.event_id` — UUID linking this pull to its update/reward
- `result.scores` — `dict[int, float]` of arm_id to score (for debugging)
- `result.bandit_id` — integer bandit ID
- `result.bandit_name` — bandit name string

### `bandito.update(pull_result, **kwargs)`

Send event data to cloud. Writes to local SQLite WAL first (crash-safe), then the background worker flushes to cloud. Returns immediately.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pull_result` | `PullResult` | Result from `pull()` |
| `query_text` | `str` | The user's query |
| `response_text` | `str` | The LLM's response |
| `reward` | `float` | Immediate reward (0.0-1.0) |
| `cost` | `float` | Cost in dollars |
| `latency` | `float` | Latency in milliseconds |
| `input_tokens` | `int` | Input token count |
| `output_tokens` | `int` | Output token count |
| `segment` | `dict[str, str]` | Key-value segment tags |

### `bandito.reward(event_id, reward, *, is_human=True)`

Send a delayed reward for an existing event. Synchronous HTTP call.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_id` | `str` | | The `event_id` from `PullResult` |
| `reward` | `float` | | Reward value (0.0-1.0) |
| `is_human` | `bool` | `True` | Whether this is a human-graded reward |

### `bandito.sync()`

Explicit state refresh from cloud. The background worker does this automatically on a schedule.

### `bandito.close()`

Shut down the background worker, attempt a final event flush, and close all connections.

## How It Works

The SDK caches the Bayesian posterior (θ, Cholesky factor) for each bandit locally. On `pull()`, it:

1. Samples `θ_tilde` from the shared posterior via Thompson Sampling
2. Builds a feature vector per arm (model one-hot + prompt one-hot + query length interaction + latency interaction)
3. Scores each arm: `score = x^T · θ_tilde`
4. Returns the highest-scoring arm

All Bayesian updates happen server-side. The SDK is a pure read cache that periodically refreshes via heartbeat.

### Crash Safety

Events are written to a local SQLite database (WAL mode) before any network call. If the process crashes, pending events are retried on the next `connect()`. No events are lost.

### Fail-Safe

If the cloud is unreachable, the SDK continues making decisions with the last-known-good weights. Your application never breaks due to a Bandito outage.

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `BANDITO_API_KEY` | API key (alternative to passing `api_key=` to `connect()`) |

## Development

```bash
cd sdk
uv sync                # Install dependencies
uv run pytest -q       # Run tests (55 tests)
```

## Architecture

```
bandito/
├── __init__.py        # Module-level API (lazy singleton)
├── client.py          # BanditoClient orchestrator
├── models.py          # Arm, PullResult, _BanditCache
├── http.py            # Sync httpx transport
├── store.py           # SQLite WAL event store
├── _worker.py         # Background sync + flush thread
└── engine/            # Pure math (copied from backend)
    ├── constants.py   # Optimization betas, reward ceilings
    ├── features.py    # ArmIdentity, ArmIndexMap, FeatureTransformer
    └── linalg.py      # sample_thompson, score_arms
```
