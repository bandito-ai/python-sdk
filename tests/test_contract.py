"""Cross-component contract tests: backend SyncResponse → SDK _apply_sync.

Validates that the JSON shape produced by the backend's _build_sync_response()
is correctly consumed by the SDK's BanditoClient._apply_sync(), with realistic
learned weights and proper dimension alignment.
"""

import numpy as np
import pytest

from bandito.client import BanditoClient
from bandito.engine import ArmIdentity, ArmIndexMap
from bandito.engine.features import compute_dimensions
from tests.conftest import ARM_DATA, EXPECTED_DIMS, make_sync_response


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_offline_client() -> BanditoClient:
    """Client with no HTTP — we'll call _apply_sync directly."""
    client = BanditoClient(api_key="bnd_test", base_url="http://unused")
    client._connected = True  # bypass connect() check for unit tests
    return client


def _make_learned_theta(dims: int, favored_model_idx: int) -> list[float]:
    """Build a theta vector that strongly favors one model.

    Sets a large positive weight on the model one-hot position so that
    Thompson Sampling consistently picks that arm regardless of noise.
    """
    theta = [0.0] * dims
    theta[favored_model_idx] = 10.0  # overwhelming signal
    return theta


# ── Contract tests ───────────────────────────────────────────────────────────


class TestSyncContract:
    """Prove the backend SyncResponse is correctly consumed by the SDK."""

    def test_nonzero_theta_exploits_learned_arm(self):
        """With theta that strongly favors one model, pull() consistently picks it.

        Simulates a backend that has learned arm 1 (gpt-4/OpenAI) is best.
        The SDK should respect those weights after _apply_sync.
        """
        # ARM_DATA: arm_id=1 is gpt-4/OpenAI, arm_id=2 is claude-sonnet/Anthropic
        # ArmIndexMap sorts by arm_id, so model index 0 = (gpt-4, OpenAI)
        d = EXPECTED_DIMS  # 8
        theta = _make_learned_theta(d, favored_model_idx=0)

        sync_data = make_sync_response([{
            "bandit_id": 1, "name": "contract-bot", "type": "online",
            "cost_importance": 0, "latency_importance": 0,
            "optimization_mode": "base", "total_pull_count": 100,
            "avg_latency_last_n": None,
            "theta": theta,
            "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [{**a, "avg_latency_last_n": None} for a in ARM_DATA],
        }])

        client = _make_offline_client()
        client._rng = np.random.default_rng(42)
        client._apply_sync(sync_data)

        # With theta[0]=10.0 and identity cholesky (noise ~N(0,1)),
        # arm 1 (gpt-4) should win every time.
        wins = {"gpt-4": 0, "claude-sonnet": 0}
        for _ in range(50):
            result = client.pull("contract-bot")
            wins[result.arm.model_name] = wins.get(result.arm.model_name, 0) + 1

        assert wins["gpt-4"] == 50, (
            f"Expected gpt-4 to win all 50 pulls with theta[0]=10.0, "
            f"but got: {wins}"
        )

    def test_dimensions_match_arm_layout(self):
        """len(theta) == 3*n_models + n_prompts and ArmIndexMap.dimensions agrees."""
        d = EXPECTED_DIMS
        sync_data = make_sync_response()

        client = _make_offline_client()
        client._apply_sync(sync_data)

        cache = client._bandits["my-chatbot"]

        # Build ArmIndexMap from the same identities the SDK built
        identities = [
            ArmIdentity(
                arm_id=a["arm_id"], model_name=a["model_name"],
                model_provider=a["model_provider"], system_prompt=a["system_prompt"],
            )
            for a in ARM_DATA
        ]
        expected_map = ArmIndexMap.from_arms(identities)

        # All three must agree: wire dimensions, cache theta length, ArmIndexMap
        assert cache.dimensions == d
        assert len(cache.theta) == d
        assert cache.chol.shape == (d, d)
        assert expected_map.dimensions == d
        assert cache.index_map.dimensions == expected_map.dimensions

        # Verify the formula: 3*n_models + n_prompts
        n_models = expected_map.n_models  # 2
        n_prompts = expected_map.n_prompts  # 2
        assert d == compute_dimensions(n_models, n_prompts)

    def test_cholesky_identity_gives_uniform_exploration(self):
        """With zero theta and identity cholesky, no arm is systematically favored.

        Pure exploration: all arms should get selected at least once over many pulls.
        """
        d = EXPECTED_DIMS
        sync_data = make_sync_response([{
            "bandit_id": 1, "name": "explore-bot", "type": "online",
            "cost_importance": 0, "latency_importance": 0,
            "optimization_mode": "base", "total_pull_count": 0,
            "avg_latency_last_n": None,
            "theta": [0.0] * d,
            "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [{**a, "avg_latency_last_n": None} for a in ARM_DATA],
        }])

        client = _make_offline_client()
        client._rng = np.random.default_rng(123)
        client._apply_sync(sync_data)

        arm_ids_seen = set()
        for _ in range(200):
            result = client.pull("explore-bot")
            arm_ids_seen.add(result.arm.arm_id)

        assert arm_ids_seen == {1, 2, 3}, (
            f"Expected all 3 arms to be selected at least once with zero theta, "
            f"but only saw arm_ids {arm_ids_seen}"
        )

    def test_response_field_coverage(self):
        """Every field _apply_sync reads exists in the mock sync response.

        Catches schema drift: if the backend adds/renames a field that the SDK
        depends on, this test will fail with a KeyError during _apply_sync.
        """
        sync_data = make_sync_response()

        # Verify top-level keys
        assert "bandits" in sync_data
        assert "server_time" in sync_data

        bandit = sync_data["bandits"][0]

        # Keys read by _apply_sync for each bandit
        required_bandit_keys = {
            "bandit_id", "name", "theta", "cholesky", "dimensions",
            "optimization_mode", "avg_latency_last_n", "arms",
        }
        assert required_bandit_keys.issubset(bandit.keys()), (
            f"Missing bandit keys: {required_bandit_keys - bandit.keys()}"
        )

        arm = bandit["arms"][0]

        # Keys read by _apply_sync for each arm
        required_arm_keys = {
            "arm_id", "model_name", "model_provider",
            "system_prompt", "is_prompt_templated", "avg_latency_last_n",
        }
        assert required_arm_keys.issubset(arm.keys()), (
            f"Missing arm keys: {required_arm_keys - arm.keys()}"
        )

        # Verify _apply_sync actually succeeds without error
        client = _make_offline_client()
        client._apply_sync(sync_data)
        assert "my-chatbot" in client._bandits

    def test_optional_fields_missing_resilience(self):
        """_apply_sync handles sync responses missing optional fields.

        Fields like budget, total_cost, avg_latency_last_n, and
        optimization_mode may be absent from the server response.
        The SDK should apply sensible defaults without error.
        """
        d = EXPECTED_DIMS
        # Minimal bandit — only required fields, no optional ones
        minimal_bandit = {
            "bandit_id": 1,
            "name": "minimal-bot",
            "theta": [0.0] * d,
            "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [{
                "arm_id": a["arm_id"],
                "model_name": a["model_name"],
                "model_provider": a["model_provider"],
                "system_prompt": a["system_prompt"],
                "is_prompt_templated": a["is_prompt_templated"],
            } for a in ARM_DATA],
            # Intentionally omitted: budget, total_cost, avg_latency_last_n,
            # optimization_mode, type, cost_importance, latency_importance,
            # total_pull_count
        }
        sync_data = {"bandits": [minimal_bandit], "server_time": "2025-01-01T00:00:00Z"}

        client = _make_offline_client()
        client._apply_sync(sync_data)

        cache = client._bandits["minimal-bot"]
        assert cache.optimization_mode == "base"  # default
        assert cache.budget is None
        assert cache.total_cost is None
        assert cache.avg_latency_last_n is None
        assert len(cache.arms) == len(ARM_DATA)

        # Should be able to pull without error
        result = client.pull("minimal-bot")
        assert result.arm is not None

    def test_feature_matrix_shape_matches_dims(self):
        """Pre-allocated feature matrix has shape (n_arms, dims)."""
        d = EXPECTED_DIMS
        sync_data = make_sync_response()

        client = _make_offline_client()
        client._apply_sync(sync_data)

        cache = client._bandits["my-chatbot"]
        assert cache.feature_matrix.shape == (len(ARM_DATA), d)
