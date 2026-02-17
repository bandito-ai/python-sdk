"""BanditoClient — main orchestrator for the SDK.

Sync-first API: no `await` anywhere in the user-facing surface.
pull() is pure local math (<1ms). update() writes to SQLite first.
Background thread handles event flush + periodic heartbeat.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from bandito._worker import BackgroundWorker
from bandito.engine import (
    DEFAULT_RELATIVE_LATENCY,
    MIN_QUERY_LENGTH,
    OPTIMIZATION_BETAS,
    ArmIdentity,
    ArmIndexMap,
    FeatureTransformer,
    sample_thompson,
    score_arms,
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


class BanditoClient:
    """Core SDK client. Sync-first, thread-safe.

    Two usage patterns:
        # Pattern 1: module-level singleton
        import bandito
        bandito.connect(api_key="bnd_...")
        result = bandito.pull("my-chatbot")

        # Pattern 2: explicit client (testing, DI)
        from bandito import BanditoClient
        client = BanditoClient(api_key="bnd_...")
        client.connect()
        result = client.pull("my-chatbot")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        sync_interval: float = 30.0,
        flush_interval: float = 5.0,
        store_path: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._sync_interval = sync_interval
        self._flush_interval = flush_interval
        self._store_path = store_path

        self._http: BanditoHTTP | None = None
        self._store: EventStore | None = None
        self._worker: BackgroundWorker | None = None
        self._bandits: dict[str, _BanditCache] = {}  # name → cache
        self._lock = threading.Lock()
        self._connected = False
        self._rng = np.random.default_rng()

    def connect(self) -> None:
        """Bootstrap: authenticate and hydrate in-memory state from cloud.

        Reads api_key from constructor arg or BANDITO_API_KEY env var.
        Creates HTTP client, SQLite store, fetches full state, starts
        background worker.
        """
        # Tear down previous connection if reconnecting
        if self._connected:
            self.close()

        api_key = self._api_key or os.environ.get("BANDITO_API_KEY")
        if not api_key:
            raise ValueError("api_key required — pass it or set BANDITO_API_KEY")

        base_url = self._base_url or os.environ.get(
            "BANDITO_BASE_URL", DEFAULT_BASE_URL
        )

        self._http = BanditoHTTP(base_url, api_key)
        store_path = self._store_path or DEFAULT_STORE_PATH
        if store_path != ":memory:":
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self._store = EventStore(store_path)

        # Fetch full state from cloud
        data = self._http.connect()
        with self._lock:
            self._apply_sync(data)

        # Flush any events pending from a previous crash
        self._flush_pending()

        # Start background worker
        self._worker = BackgroundWorker(
            self._http,
            self._store,
            self._on_sync,
            sync_interval=self._sync_interval,
            flush_interval=self._flush_interval,
        )
        self._worker.start()
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
        response_text: str | None = None,
        reward: float | None = None,
        cost: float | None = None,
        latency: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        segment: dict[str, str] | None = None,
    ) -> None:
        """Send event data to cloud. Writes to SQLite first (crash-safe).

        Args:
            pull_result: Result from pull().
            query_text: The user's query text.
            response_text: The LLM's response text.
            reward: Immediate reward (0.0-1.0).
            cost: Cost in dollars.
            latency: Latency in milliseconds.
            input_tokens: Input token count.
            output_tokens: Output token count.
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
            event["response_text"] = response_text
        if reward is not None:
            event["immediate_reward"] = reward
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

        # Wake background worker for immediate flush
        if self._worker:
            self._worker.trigger_flush()

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
        """Shut down worker, flush remaining events, close connections."""
        if self._worker:
            self._worker.stop()
            self._worker = None

        # Final flush attempt
        if self._store and self._http:
            self._flush_pending()

        if self._store:
            self._store.close()
            self._store = None
        if self._http:
            self._http.close()
            self._http = None
        self._connected = False

    # ── Internal ──────────────────────────────────────────────────────

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
            transformer = FeatureTransformer(index_map)

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
                transformer=transformer,
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

    def _on_sync(self, data: dict[str, Any]) -> None:
        """Callback from background worker when heartbeat succeeds."""
        with self._lock:
            self._apply_sync(data)

    def _flush_pending(self) -> None:
        """Attempt to flush pending SQLite events to cloud."""
        try:
            pending = self._store.pending()
            if not pending:
                return
            self._http.ingest_events(pending)
            uuids = [e["local_event_uuid"] for e in pending]
            self._store.mark_flushed(uuids)
            logger.debug("Flushed %d pending events on connect/close", len(pending))
        except Exception:
            logger.warning("Failed to flush pending events", exc_info=True)
