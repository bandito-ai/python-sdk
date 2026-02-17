"""Tests for the engine modules copied from backend."""

import numpy as np
import pytest

from bandito.engine import (
    ArmIdentity,
    ArmIndexMap,
    FeatureTransformer,
    sample_thompson,
    score_arms,
)
from bandito.engine.constants import OPTIMIZATION_BETAS
from tests.conftest import EXPECTED_DIMS


class TestArmIndexMap:
    def test_from_arms_dimensions(self, arm_identities):
        idx = ArmIndexMap.from_arms(arm_identities)
        assert idx.dimensions == EXPECTED_DIMS
        assert idx.n_models == 2
        assert idx.n_prompts == 2

    def test_from_arms_deterministic(self, arm_identities):
        """Same arms in different order produce same index map."""
        idx1 = ArmIndexMap.from_arms(arm_identities)
        idx2 = ArmIndexMap.from_arms(list(reversed(arm_identities)))
        assert idx1.model_to_index == idx2.model_to_index
        assert idx1.prompt_to_index == idx2.prompt_to_index

    def test_from_arms_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ArmIndexMap.from_arms([])


class TestFeatureTransformer:
    def test_transform_shape(self, transformer, arm_identities):
        x = transformer.transform(arm_identities[0])
        assert x.shape == (EXPECTED_DIMS,)
        assert x.dtype == np.float64

    def test_transform_cold_start_defaults(self, transformer, arm_identities):
        """Without query/latency, interaction blocks should be neutral."""
        x = transformer.transform(arm_identities[0])
        # log(1) = 0.0 for query block, 1.0 for latency block
        m = transformer._map
        log_ql_idx = m.n_models + m.n_prompts + m.model_to_index[("gpt-4", "OpenAI")]
        assert x[log_ql_idx] == 0.0  # log(1) = 0
        rel_lat_idx = 2 * m.n_models + m.n_prompts + m.model_to_index[("gpt-4", "OpenAI")]
        assert x[rel_lat_idx] == 1.0  # DEFAULT_RELATIVE_LATENCY


class TestSampling:
    def test_sample_thompson_shape(self, identity_theta):
        theta, chol = identity_theta
        rng = np.random.default_rng(42)
        result = sample_thompson(theta, chol, beta=1.0, rng=rng)
        assert result.shape == theta.shape

    def test_score_arms_ranking(self):
        """Higher-weighted arm should score higher."""
        theta_tilde = np.array([2.0, 0.5])
        vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        scores = score_arms(theta_tilde, vecs)
        assert scores[0] > scores[1]  # arm 0 aligned with weight=2.0
