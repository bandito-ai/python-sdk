"""Shared fixtures for SDK tests."""

import numpy as np
import pytest

from bandito.engine import ArmIdentity, ArmIndexMap, FeatureTransformer
from bandito.models import Arm


# ── Standard 3-arm bandit setup ──────────────────────────────────────────

ARM_DATA = [
    {"arm_id": 1, "model_name": "gpt-4", "model_provider": "OpenAI", "system_prompt": "You are helpful", "is_prompt_templated": False},
    {"arm_id": 2, "model_name": "claude-sonnet", "model_provider": "Anthropic", "system_prompt": "You are helpful", "is_prompt_templated": False},
    {"arm_id": 3, "model_name": "gpt-4", "model_provider": "OpenAI", "system_prompt": "Be concise", "is_prompt_templated": True},
]

# 2 models, 2 prompts → dims = 3*2 + 2 = 8
EXPECTED_DIMS = 8


@pytest.fixture
def arm_identities():
    return [
        ArmIdentity(arm_id=a["arm_id"], model_name=a["model_name"],
                     model_provider=a["model_provider"], system_prompt=a["system_prompt"])
        for a in ARM_DATA
    ]


@pytest.fixture
def arms():
    return [Arm(**a) for a in ARM_DATA]


@pytest.fixture
def index_map(arm_identities):
    return ArmIndexMap.from_arms(arm_identities)


@pytest.fixture
def transformer(index_map):
    return FeatureTransformer(index_map)


@pytest.fixture
def identity_theta():
    """Cold-start theta (zeros) and chol (identity) for 8 dims."""
    d = EXPECTED_DIMS
    return np.zeros(d), np.eye(d)


def make_sync_response(bandits=None, *, budget=None, total_cost=None):
    """Build a mock sync response matching the backend SyncResponse schema."""
    if bandits is None:
        d = EXPECTED_DIMS
        bandits = [{
            "bandit_id": 1,
            "name": "my-chatbot",
            "type": "online",
            "cost_importance": 2,
            "latency_importance": 3,
            "optimization_mode": "base",
            "total_pull_count": 0,
            "avg_latency_last_n": None,
            "budget": budget,
            "total_cost": total_cost,
            "theta": [0.0] * d,
            "cholesky": np.eye(d).tolist(),
            "dimensions": d,
            "arms": [
                {**a, "avg_latency_last_n": None}
                for a in ARM_DATA
            ],
        }]
    return {
        "bandits": bandits,
        "server_time": "2025-01-01T00:00:00Z",
    }
