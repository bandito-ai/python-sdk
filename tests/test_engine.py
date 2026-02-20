"""Tests for the engine modules copied from backend."""

import numpy as np
import pytest

from bandito.engine import (
    ArmIdentity,
    ArmIndexMap,
    sample_thompson,
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


class TestSampling:
    def test_sample_thompson_shape(self, identity_theta):
        theta, chol = identity_theta
        rng = np.random.default_rng(42)
        result = sample_thompson(theta, chol, beta=1.0, rng=rng)
        assert result.shape == theta.shape
