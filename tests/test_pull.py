"""Tests for the pull() method — local Thompson Sampling decisions."""

import math

import numpy as np
import pytest
import httpx
import respx

from bandito.client import BanditoClient
from bandito.engine import (
    ArmIdentity,
    ArmIndexMap,
    DEFAULT_RELATIVE_LATENCY,
    FeatureTransformer,
    MIN_QUERY_LENGTH,
    score_arms,
)
from bandito.models import PullResult
from tests.conftest import ARM_DATA, EXPECTED_DIMS, make_sync_response


BASE_URL = "http://test.local"
API_KEY = "bnd_test123"


def _connected_client(sync_data=None) -> BanditoClient:
    """Create a client that's already connected with mocked HTTP."""
    if sync_data is None:
        sync_data = make_sync_response()
    respx.post(f"{BASE_URL}/api/v1/sync/connect").mock(
        return_value=httpx.Response(200, json=sync_data)
    )
    client = BanditoClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        flush_interval=9999,
        sync_interval=9999,
        store_path=":memory:",
    )
    client.connect()
    return client


class TestPull:
    @respx.mock
    def test_pull_returns_pull_result(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            assert isinstance(result, PullResult)
            assert result.bandit_name == "my-chatbot"
            assert result.bandit_id == 1
            assert result.arm is not None
            assert result.event_id  # non-empty UUID string
        finally:
            client.close()

    @respx.mock
    def test_pull_event_id_unique(self):
        client = _connected_client()
        try:
            r1 = client.pull("my-chatbot")
            r2 = client.pull("my-chatbot")
            assert r1.event_id != r2.event_id
        finally:
            client.close()

    @respx.mock
    def test_pull_scores_all_arms(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            assert len(result.scores) == 3
            assert set(result.scores.keys()) == {1, 2, 3}
        finally:
            client.close()

    @respx.mock
    def test_pull_winner_has_highest_score(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            winner_score = result.scores[result.arm.arm_id]
            assert winner_score == max(result.scores.values())
        finally:
            client.close()

    @respx.mock
    def test_pull_unknown_bandit_raises(self):
        client = _connected_client()
        try:
            with pytest.raises(KeyError, match="Unknown bandit 'nope'"):
                client.pull("nope")
        finally:
            client.close()

    @respx.mock
    def test_pull_with_query(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot", query="What is 2+2?")
            assert isinstance(result, PullResult)
        finally:
            client.close()

    @respx.mock
    def test_pull_not_connected_raises(self):
        client = BanditoClient(api_key="x")
        with pytest.raises(RuntimeError, match="Not connected"):
            client.pull("test")

    @respx.mock
    def test_pull_convenience_properties(self):
        client = _connected_client()
        try:
            result = client.pull("my-chatbot")
            # model/prompt should match arm data
            assert result.model == result.arm.model_name
            assert result.prompt == result.arm.system_prompt
        finally:
            client.close()

    @respx.mock
    def test_pull_explore_mode(self):
        """Explore mode bandit should work without errors."""
        d = EXPECTED_DIMS
        explore_sync = make_sync_response([{
            "bandit_id": 1, "name": "explore-bot", "type": "online",
            "cost_importance": 0, "latency_importance": 0,
            "optimization_mode": "explore",
            "total_pull_count": 0, "avg_latency_last_n": None,
            "theta": [0.0] * d, "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [{**a, "avg_latency_last_n": None} for a in ARM_DATA],
        }])
        client = _connected_client(explore_sync)
        try:
            result = client.pull("explore-bot")
            assert isinstance(result, PullResult)
        finally:
            client.close()

    @respx.mock
    def test_pull_with_latency_context(self):
        """Arms with latency data should compute relative_latency."""
        d = EXPECTED_DIMS
        sync = make_sync_response([{
            "bandit_id": 1, "name": "latency-bot", "type": "online",
            "cost_importance": 0, "latency_importance": 3,
            "optimization_mode": "base",
            "total_pull_count": 100, "avg_latency_last_n": 1000.0,
            "theta": [0.0] * d, "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [
                {**ARM_DATA[0], "avg_latency_last_n": 800.0},
                {**ARM_DATA[1], "avg_latency_last_n": 1200.0},
                {**ARM_DATA[2], "avg_latency_last_n": 1000.0},
            ],
        }])
        client = _connected_client(sync)
        try:
            result = client.pull("latency-bot")
            assert isinstance(result, PullResult)
        finally:
            client.close()


# ---------- Feature Matrix Equivalence ----------


class TestFeatureMatrixEquivalence:
    """Prove the optimized pre-allocated feature matrix produces identical
    vectors to the original FeatureTransformer.transform() path."""

    QUERIES = [None, "Hi", "What is the meaning of life?" * 10]

    LATENCY_CONFIGS = [
        # (bandit_avg, arm_latencies) → arm_avg_latencies dict built per test
        (None, {1: None, 2: None, 3: None}),
        (1000.0, {1: 800.0, 2: 1200.0, 3: 1000.0}),
        (500.0, {1: None, 2: 750.0, 3: 250.0}),
    ]

    def _build_reference_vectors(
        self, identities, transformer, query, bandit_avg, arm_latencies,
    ):
        """Original path: per-arm transform() → list of vectors."""
        query_length = len(query) if query else None
        vecs = []
        for identity in identities:
            arm_lat = arm_latencies.get(identity.arm_id)
            if arm_lat and bandit_avg and bandit_avg > 0:
                rel = arm_lat / bandit_avg
            else:
                rel = None
            vecs.append(transformer.transform(
                identity, query_length=query_length, relative_latency=rel,
            ))
        return np.array(vecs)

    def _build_optimized_matrix(
        self, identities, index_map, query, bandit_avg, arm_latencies,
    ):
        """Optimized path: pre-allocated matrix with one-hot, overwrite context."""
        dims = index_map.dimensions
        X = np.zeros((len(identities), dims), dtype=np.float64)

        # Static one-hot (filled at sync time)
        for i, identity in enumerate(identities):
            model_idx = index_map.model_to_index[(identity.model_name, identity.model_provider)]
            prompt_idx = index_map.prompt_to_index[identity.system_prompt]
            X[i, model_idx] = 1.0
            X[i, index_map.n_models + prompt_idx] = 1.0

        # Context columns (filled at pull time)
        query_length = len(query) if query else None
        ql = max(query_length or MIN_QUERY_LENGTH, MIN_QUERY_LENGTH)
        log_ql = math.log(ql)

        for i, identity in enumerate(identities):
            model_idx = index_map.model_to_index[(identity.model_name, identity.model_provider)]
            X[i, index_map.n_models + index_map.n_prompts + model_idx] = log_ql

            arm_lat = arm_latencies.get(identity.arm_id)
            if arm_lat and bandit_avg and bandit_avg > 0:
                rl = arm_lat / bandit_avg
            else:
                rl = DEFAULT_RELATIVE_LATENCY
            X[i, 2 * index_map.n_models + index_map.n_prompts + model_idx] = rl

        return X

    def test_feature_vectors_identical(self, arm_identities, index_map, transformer):
        """Every query x latency combo produces identical feature matrices."""
        for query in self.QUERIES:
            for bandit_avg, arm_lats in self.LATENCY_CONFIGS:
                ref = self._build_reference_vectors(
                    arm_identities, transformer, query, bandit_avg, arm_lats,
                )
                opt = self._build_optimized_matrix(
                    arm_identities, index_map, query, bandit_avg, arm_lats,
                )
                np.testing.assert_array_equal(
                    ref, opt,
                    err_msg=f"Mismatch for query={query!r}, latency={bandit_avg}",
                )

    def test_scores_identical(self, arm_identities, index_map, transformer):
        """Scores from both paths match for a non-trivial theta."""
        rng = np.random.default_rng(42)
        theta = rng.standard_normal(index_map.dimensions)

        for query in self.QUERIES:
            for bandit_avg, arm_lats in self.LATENCY_CONFIGS:
                ref = self._build_reference_vectors(
                    arm_identities, transformer, query, bandit_avg, arm_lats,
                )
                opt = self._build_optimized_matrix(
                    arm_identities, index_map, query, bandit_avg, arm_lats,
                )
                ref_scores = score_arms(theta, list(ref))
                opt_scores = opt @ theta
                np.testing.assert_allclose(
                    ref_scores, opt_scores, atol=1e-15,
                    err_msg=f"Score mismatch for query={query!r}, latency={bandit_avg}",
                )
