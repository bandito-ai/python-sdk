"""BanditoClient — main orchestrator for the SDK.

Sync-first API: no ``await`` anywhere in the user-facing surface.
``pull()`` is pure local math (<1ms). ``update()`` writes to SQLite first,
then submits a non-blocking flush to a single-threaded executor.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from bandito._worker import prepare_cloud_payload
from bandito.engine import (
    DEFAULT_RELATIVE_LATENCY,
    MIN_QUERY_LENGTH,
    OPTIMIZATION_BETAS,
    ArmIdentity,
    ArmIndexMap,
    sample_thompson,
)
from bandito.config import DEFAULT_BASE_URL
from bandito.http import BanditoHTTP
from bandito.models import Arm, PullResult, _BanditCache
from bandito.store import EventStore

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("bandito")
except Exception:
    __version__ = "0.1.0"

logger = logging.getLogger("bandito")
logger.addHandler(logging.NullHandler())

DEFAULT_STORE_PATH = str(Path.home() / ".bandito" / "events.db")
_MAX_EVENT_RETRIES = 5  # after this many server rejections, skip the event


class BanditoClient:
    """Core SDK client. Sync-first, thread-safe.

    Recommended usage (context manager):

        from bandito import BanditoClient

        with BanditoClient(api_key="bnd_...") as client:
            result = client.pull("my-chatbot", query=user_message)
            response = call_llm(result.model, result.prompt, user_message)
            client.update(result, response_text=response.text)

    Explicit connect/close:

        client = BanditoClient(api_key="bnd_...")
        client.connect()
        ...
        client.close()

    API key resolution order: constructor arg -> BANDITO_API_KEY env var
    -> ~/.bandito/config.toml (written by ``bandito init``).

    ``data_storage`` controls whether query/response text is sent to the
    cloud API. Resolution: constructor arg -> config.toml -> default "local".
    Text is always stored in local SQLite regardless of this setting.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        store_path: str | None = None,
        data_storage: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._store_path = store_path
        self._data_storage_arg = data_storage

        self._http: BanditoHTTP | None = None
        self._store: EventStore | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._bandits: dict[str, _BanditCache] = {}  # name -> cache
        self._lock = threading.Lock()
        self._connected = False
        self._data_storage = data_storage or "local"
        self._rng = np.random.default_rng()
        self._dead_uuids: set[str] = set()  # events permanently rejected by server
        self._retry_counts: dict[str, int] = {}  # uuid -> rejection count

    def __enter__(self) -> BanditoClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def connect(self) -> None:
        """Bootstrap: authenticate and hydrate in-memory state from cloud.

        Reads api_key from constructor arg -> BANDITO_API_KEY env var ->
        ~/.bandito/config.toml. Creates HTTP client, SQLite store, fetches
        full state, and creates the background flush executor.
        """
        # Tear down previous connection if reconnecting
        if self._connected:
            self.close()

        # Resolve config: constructor arg -> env var -> config.toml -> default
        from bandito.config import load_config
        config = load_config()

        api_key = self._api_key or config.api_key
        if not api_key:
            raise ValueError(
                "api_key required — pass it to connect(), set BANDITO_API_KEY, "
                "or run `bandito init`"
            )

        base_url = self._base_url or config.base_url

        if not self._data_storage_arg:
            self._data_storage = config.data_storage

        self._http = BanditoHTTP(base_url, api_key)
        store_path = self._store_path or DEFAULT_STORE_PATH
        if store_path != ":memory:":
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self._store = EventStore(store_path)

        # Fetch full state from cloud
        data = self._http.connect()
        with self._lock:
            self._apply_sync(data)

        # Reset retry state — reconnect gives previously-rejected events another chance
        self._dead_uuids.clear()
        self._retry_counts.clear()

        # Flush any events pending from a previous crash
        self._flush_pending()

        # Create executor for non-blocking event flushes.
        # Python's atexit automatically calls shutdown(wait=True) on live
        # executors, so pending flushes complete before process exit.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._connected = True

        logger.info("Connected — %d bandits", len(self._bandits))

    def pull(
        self,
        bandit_name: str,
        *,
        query: str | None = None,
    ) -> PullResult:
        """Local Thompson Sampling decision. Pure math, <1ms, no network.

        Args:
            bandit_name: Name of the bandit to pull from.
            query: User query text (used for feature engineering).

        Returns:
            PullResult with the winning arm and event_id.
        """
        self._ensure_connected()

        with self._lock:
            cache = self._bandits.get(bandit_name)
            if cache is None:
                available = list(self._bandits.keys())
                raise KeyError(
                    f"Unknown bandit '{bandit_name}'. "
                    f"Available: {available}"
                )

            if not cache.arms:
                raise ValueError(f"Bandit '{bandit_name}' has no active arms")

            # Sample from shared posterior (reuse cached RNG)
            beta = OPTIMIZATION_BETAS.get(cache.optimization_mode, 1.0)
            theta_tilde = sample_thompson(
                cache.theta, cache.chol, beta, rng=self._rng,
            )

            # Overwrite context columns in pre-allocated feature matrix.
            # Static one-hot blocks (model, prompt) are set once at sync time.
            X = cache.feature_matrix
            m = cache.index_map
            query_length = len(query) if query else None
            ql = max(query_length or MIN_QUERY_LENGTH, MIN_QUERY_LENGTH)
            log_ql = math.log(ql)

            for i, identity in enumerate(cache.arm_identities):
                model_idx = m.model_to_index[(identity.model_name, identity.model_provider)]

                # Block 3: log(query_length) * model
                X[i, m.n_models + m.n_prompts + model_idx] = log_ql

                # Block 4: relative_latency * model
                arm_latency = cache.arm_avg_latencies.get(identity.arm_id)
                bandit_latency = cache.avg_latency_last_n
                if arm_latency and bandit_latency and bandit_latency > 0:
                    rl = arm_latency / bandit_latency
                else:
                    rl = DEFAULT_RELATIVE_LATENCY
                X[i, 2 * m.n_models + m.n_prompts + model_idx] = rl

            # Score: X @ theta_tilde (no array copy needed)
            scores_array = X @ theta_tilde

            # Map scores to arm_id
            scores = {
                cache.arm_identities[i].arm_id: float(scores_array[i])
                for i in range(len(cache.arm_identities))
            }

            # Winner = highest score
            winner_idx = int(np.argmax(scores_array))
            winner_arm = cache.arms[winner_idx]

        return PullResult(
            arm=winner_arm,
            event_id=str(uuid.uuid4()),
            bandit_id=cache.bandit_id,
            bandit_name=bandit_name,
            scores=scores,
        )

    def update(
        self,
        pull_result: PullResult,
        *,
        query_text: str | None = None,
        response_text: str | dict | None = None,  # TODO: rename to `model_response` — not always text
        reward: float | None = None,
        cost: float | None = None,
        latency: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        segment: dict[str, str] | None = None,
    ) -> None:
        """Record an LLM call outcome. Writes to SQLite first (crash-safe),
        then submits a non-blocking flush to the background executor.

        Cost handling: if ``cost`` is omitted but both ``input_tokens`` and
        ``output_tokens`` are provided, the server auto-calculates cost from
        the arm's model pricing. An explicit ``cost`` always wins.

        Text storage: ``query_text`` and ``response_text`` are always saved
        to local SQLite (for TUI grading). Whether they are also sent to
        the cloud depends on the ``data_storage`` setting ("local" keeps
        them local-only; "cloud" sends them).

        For delayed or human-graded rewards, use ``bandito.reward()`` instead
        of the ``reward`` parameter here.

        Args:
            pull_result: Result from pull().
            query_text: The user's query text.
            response_text: The LLM's response text. Accepts a string or
                dict. Strings are normalized to ``{"response": "..."}``
                before storage.
            reward: Immediate reward (0.0-1.0).
            cost: Cost in dollars. Omit to let the server auto-calculate
                from token counts.
            latency: Latency in milliseconds.
            input_tokens: Input token count (enables auto-cost when cost
                is omitted).
            output_tokens: Output token count (enables auto-cost when cost
                is omitted).
            segment: Key-value segment tags.
        """
        self._ensure_connected()

        event: dict[str, Any] = {
            "local_event_uuid": pull_result.event_id,
            "bandit_id": pull_result.bandit_id,
            "arm_id": pull_result.arm.arm_id,
            "model_name": pull_result.arm.model_name,
            "model_provider": pull_result.arm.model_provider,
        }
        if query_text is not None:
            event["query_text"] = query_text
        if response_text is not None:
            if isinstance(response_text, str):
                event["response_text"] = {"response": response_text}
            else:
                event["response_text"] = response_text
        if reward is not None:
            event["immediate_reward"] = reward  # backend schema field name
        if cost is not None:
            event["cost"] = cost
        if latency is not None:
            event["latency"] = latency
        if input_tokens is not None:
            event["input_tokens"] = input_tokens
        if output_tokens is not None:
            event["output_tokens"] = output_tokens
        if segment is not None:
            event["segment"] = segment

        # Write to SQLite WAL first — survives crashes
        self._store.push(event)

        # Submit non-blocking flush to executor
        if self._executor:
            self._executor.submit(self._flush_pending)

    def reward(
        self,
        event_id: str,
        reward: float,
        *,
        is_human: bool = True,
    ) -> None:
        """Send a delayed reward for an existing event.

        This is synchronous HTTP — the user expects confirmation.

        Args:
            event_id: The event_id from PullResult.
            reward: Reward value (0.0-1.0).
            is_human: Whether this is a human-graded reward.
        """
        self._ensure_connected()
        self._http.update_reward(event_id, reward, is_human_reward=is_human)

    def sync(self) -> None:
        """Explicit state refresh from cloud."""
        self._ensure_connected()
        data = self._http.heartbeat()
        with self._lock:
            self._apply_sync(data)
        logger.info("Manual sync — %d bandits", len(self._bandits))

    def close(self) -> None:
        """Shut down executor, flush remaining events, close connections."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Final synchronous flush — catches anything the last submit missed
        if self._store and self._http:
            self._flush_pending()

        if self._store:
            self._store.close()
            self._store = None
        if self._http:
            self._http.close()
            self._http = None
        self._connected = False

    # -- Internal ----------------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected — call connect() first")

    def _apply_sync(self, data: dict[str, Any]) -> None:
        """Hydrate _bandits cache from sync response. Caller holds lock."""
        self._bandits.clear()

        for b in data.get("bandits", []):
            arms: list[Arm] = []
            identities: list[ArmIdentity] = []
            arm_latencies: dict[int, float | None] = {}

            for a in b.get("arms", []):
                arms.append(Arm(
                    arm_id=a["arm_id"],
                    model_name=a["model_name"],
                    model_provider=a["model_provider"],
                    system_prompt=a["system_prompt"],
                    is_prompt_templated=a["is_prompt_templated"],
                ))
                identities.append(ArmIdentity(
                    arm_id=a["arm_id"],
                    model_name=a["model_name"],
                    model_provider=a["model_provider"],
                    system_prompt=a["system_prompt"],
                ))
                arm_latencies[a["arm_id"]] = a.get("avg_latency_last_n")

            if not identities:
                continue

            index_map = ArmIndexMap.from_arms(identities)

            theta_raw = b["theta"]
            chol_raw = b["cholesky"]

            # Pre-allocate feature matrix with static one-hot blocks filled.
            # Context columns (log_query_len, rel_latency) are overwritten per pull().
            dims = index_map.dimensions
            n_arms = len(identities)
            feature_matrix = np.zeros((n_arms, dims), dtype=np.float64)
            for i, identity in enumerate(identities):
                model_idx = index_map.model_to_index[(identity.model_name, identity.model_provider)]
                prompt_idx = index_map.prompt_to_index[identity.system_prompt]
                feature_matrix[i, model_idx] = 1.0
                feature_matrix[i, index_map.n_models + prompt_idx] = 1.0

            self._bandits[b["name"]] = _BanditCache(
                bandit_id=b["bandit_id"],
                name=b["name"],
                theta=np.array(theta_raw, dtype=np.float64),
                chol=np.array(chol_raw, dtype=np.float64),
                dimensions=b["dimensions"],
                optimization_mode=b.get("optimization_mode", "base"),
                arms=arms,
                arm_identities=identities,
                index_map=index_map,
                avg_latency_last_n=b.get("avg_latency_last_n"),
                arm_avg_latencies=arm_latencies,
                budget=b.get("budget"),
                total_cost=b.get("total_cost"),
                feature_matrix=feature_matrix,
            )

        for name, cache in self._bandits.items():
            self._check_budget(name, cache)

    def _check_budget(self, bandit_name: str, cache: _BanditCache) -> None:
        """Log warnings when spend approaches or exceeds budget."""
        if cache.budget is None or cache.budget <= 0 or cache.total_cost is None:
            return
        ratio = cache.total_cost / cache.budget
        pct = ratio * 100
        if ratio >= 1.10:
            logger.warning(
                "Bandit '%s' has EXCEEDED budget: $%.2f / $%.2f (%.0f%%)",
                bandit_name, cache.total_cost, cache.budget, pct,
            )
        elif ratio >= 1.00:
            logger.warning(
                "Bandit '%s' has reached budget: $%.2f / $%.2f (%.0f%%)",
                bandit_name, cache.total_cost, cache.budget, pct,
            )
        elif ratio >= 0.90:
            logger.warning(
                "Bandit '%s' is approaching budget: $%.2f / $%.2f (%.0f%%)",
                bandit_name, cache.total_cost, cache.budget, pct,
            )

    def _flush_pending(self) -> None:
        """Attempt to flush pending SQLite events to cloud.

        Handles partial acceptance: events the server accepted or
        deduplicated are marked flushed. Events the server rejected are
        retried up to ``_MAX_EVENT_RETRIES`` times, then marked as dead
        (skipped on future flushes until reconnect).
        """
        try:
            pending = self._store.pending()
            if not pending:
                return

            # Skip events already known to be permanently rejected
            if self._dead_uuids:
                pending = [e for e in pending if e["local_event_uuid"] not in self._dead_uuids]
                if not pending:
                    return

            payload = prepare_cloud_payload(pending, include_text=(self._data_storage != "local"))
            logger.debug("Flush payload: %s", payload)
            result = self._http.ingest_events(payload)

            # Parse per-event errors from server response
            errored_uuids: set[str] = set()
            for err in result.get("errors", []):
                uid = err.get("local_event_uuid")
                if uid:
                    errored_uuids.add(uid)
                    count = self._retry_counts.get(uid, 0) + 1
                    self._retry_counts[uid] = count
                    if count >= _MAX_EVENT_RETRIES:
                        self._dead_uuids.add(uid)
                        logger.warning(
                            "Event %s permanently rejected after %d attempts: %s",
                            uid, count, err.get("reason", "unknown"),
                        )
                    else:
                        logger.debug(
                            "Event %s rejected (attempt %d/%d): %s",
                            uid, count, _MAX_EVENT_RETRIES, err.get("reason", "unknown"),
                        )

            # Mark accepted + deduplicated events as flushed
            flushed_uuids = [
                e["local_event_uuid"] for e in pending
                if e["local_event_uuid"] not in errored_uuids
            ]
            if flushed_uuids:
                self._store.mark_flushed(flushed_uuids)
            logger.debug(
                "Flushed %d events (errors=%d, dead=%d)",
                len(flushed_uuids), len(errored_uuids), len(self._dead_uuids),
            )
        except Exception:
            logger.warning("Failed to flush pending events", exc_info=True)
