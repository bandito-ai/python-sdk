"""
Feature engineering for shared Linear Thompson Sampling.

Pure module — imports only numpy and constants. Safe for SDK and server.

All arms in a bandit share ONE posterior (theta, A, b). Arms are differentiated
by feature engineering: each arm gets a unique feature vector encoding its
identity (one-hot) plus context interaction terms.

Feature vector layout for an arm with M unique models and P unique prompts:

    [model_one_hot(M) | prompt_one_hot(P) | log_query_len * model(M) | rel_latency * model(M)]
     ─── M dims ───   ─── P dims ───      ─── M dims ───              ─── M dims ───
    Total dimensions = 3M + P

Why no global bias term:
    The one-hot blocks already absorb per-arm baselines. A global intercept
    would be collinear with the sum of one-hot columns.

Cold-start defaults:
    - query_length=1 → log(1)=0.0, so the interaction block is all zeros (neutral)
    - relative_latency=1.0 → "average" latency, no directional signal

Extension guide:
    1. Add a new feature block in FeatureTransformer.transform()
    2. Update compute_dimensions() to account for the new block size
    3. Add names in FeatureTransformer.get_feature_names()
    4. BanditState.resize() handles matrix growth automatically
"""

from __future__ import annotations

from dataclasses import dataclass, field



@dataclass(frozen=True)
class ArmIdentity:
    """Minimal arm representation decoupled from ORM.

    Server builds this from the Arm model. SDK builds from sync payload.
    Same type, different sources — that's the portability boundary.

    Args:
        arm_id: Unique integer identifier for the arm.
        model_name: LLM model identifier (e.g., "claude-sonnet-4-5-20250514").
        model_provider: Provider name (e.g., "Anthropic", "Bedrock").
        system_prompt: The system prompt text (used for identity, not content).
    """
    arm_id: int
    model_name: str
    model_provider: str
    system_prompt: str


def compute_dimensions(n_models: int, n_prompts: int) -> int:
    """Compute total feature vector dimensionality.

    dims = 3 * n_models + n_prompts

    The three model blocks are: model identity, log(query_len) interaction,
    and relative_latency interaction.

    Args:
        n_models: Number of unique (model_name, model_provider) pairs.
        n_prompts: Number of unique system prompts.

    Returns:
        Total feature dimensions.

    Example:
        >>> compute_dimensions(n_models=3, n_prompts=2)
        11  # 3*3 + 2
    """
    return 3 * n_models + n_prompts


@dataclass(frozen=True)
class ArmIndexMap:
    """Maps arm identities to one-hot indices for feature construction.

    Built once per bandit configuration. Sorting by arm_id ensures deterministic
    ordering regardless of DB query order or insertion sequence.

    Attributes:
        model_to_index: Maps (model_name, model_provider) → column index in model block.
        prompt_to_index: Maps system_prompt → column index in prompt block.
        n_models: Number of unique models.
        n_prompts: Number of unique prompts.
        dimensions: Total feature vector length (3*n_models + n_prompts).
    """
    model_to_index: dict[tuple[str, str], int] = field(default_factory=dict)
    prompt_to_index: dict[str, int] = field(default_factory=dict)
    n_models: int = 0
    n_prompts: int = 0
    dimensions: int = 0

    @classmethod
    def from_arms(cls, arms: list[ArmIdentity]) -> ArmIndexMap:
        """Build index mappings from a list of arm identities.

        Arms are sorted by arm_id to guarantee deterministic index assignment.
        This is critical: the same set of arms must always produce the same
        feature layout, or the learned theta becomes meaningless.

        Args:
            arms: List of ArmIdentity objects.

        Returns:
            Frozen ArmIndexMap ready for FeatureTransformer.

        Raises:
            ValueError: If arms list is empty.

        Example:
            >>> arms = [
            ...     ArmIdentity(1, "gpt-4", "OpenAI", "You are helpful"),
            ...     ArmIdentity(2, "claude-sonnet", "Anthropic", "You are helpful"),
            ...     ArmIdentity(3, "gpt-4", "OpenAI", "Be concise"),
            ... ]
            >>> idx = ArmIndexMap.from_arms(arms)
            >>> idx.n_models, idx.n_prompts, idx.dimensions
            (2, 2, 8)  # 3*2 + 2
        """
        if not arms:
            raise ValueError("Cannot build index map from empty arm list")

        sorted_arms = sorted(arms, key=lambda a: a.arm_id)

        models: dict[tuple[str, str], int] = {}
        prompts: dict[str, int] = {}

        for arm in sorted_arms:
            key = (arm.model_name, arm.model_provider)
            if key not in models:
                models[key] = len(models)
            if arm.system_prompt not in prompts:
                prompts[arm.system_prompt] = len(prompts)

        n_m = len(models)
        n_p = len(prompts)
        return cls(
            model_to_index=models,
            prompt_to_index=prompts,
            n_models=n_m,
            n_prompts=n_p,
            dimensions=compute_dimensions(n_m, n_p),
        )


