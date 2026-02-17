"""SDK types: Arm, PullResult, and internal cache structures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from bandito.engine import ArmIdentity, ArmIndexMap, FeatureTransformer


@dataclass(frozen=True)
class Arm:
    """An arm returned to the user after pull(). Frozen for safety."""

    arm_id: int
    model_name: str
    model_provider: str
    system_prompt: str
    is_prompt_templated: bool

    @property
    def model(self) -> str:
        """Convenience alias for model_name."""
        return self.model_name

    @property
    def prompt(self) -> str:
        """Convenience alias for system_prompt."""
        return self.system_prompt


@dataclass(frozen=True)
class PullResult:
    """Returned by pull(), passed to update(). Frozen."""

    arm: Arm
    event_id: str
    bandit_id: int
    bandit_name: str
    scores: dict[int, float]

    @property
    def model(self) -> str:
        """Reach-through to arm.model_name."""
        return self.arm.model_name

    @property
    def prompt(self) -> str:
        """Reach-through to arm.system_prompt."""
        return self.arm.system_prompt


@dataclass
class _BanditCache:
    """Internal mutable cache for a bandit's Bayesian state."""

    bandit_id: int
    name: str
    theta: np.ndarray
    chol: np.ndarray
    dimensions: int
    optimization_mode: str
    arms: list[Arm]
    arm_identities: list[ArmIdentity]
    index_map: ArmIndexMap
    transformer: FeatureTransformer
    avg_latency_last_n: float | None
    arm_avg_latencies: dict[int, float | None]
    budget: float | None = None
    total_cost: float | None = None
    # Pre-allocated feature matrix (n_arms x dims). Static one-hot blocks are
    # filled once at sync time; context columns are overwritten each pull().
    feature_matrix: np.ndarray = field(default=None)  # type: ignore[assignment]
