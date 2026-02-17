"""Pure math engine for Linear Thompson Sampling.

Copied from backend/app/services/bandit/ — zero app imports, SDK-portable.
"""

from .constants import DEFAULT_RELATIVE_LATENCY, MIN_QUERY_LENGTH, OPTIMIZATION_BETAS
from .features import ArmIdentity, ArmIndexMap, FeatureTransformer
from .linalg import sample_thompson, score_arms

__all__ = [
    "OPTIMIZATION_BETAS",
    "DEFAULT_RELATIVE_LATENCY",
    "MIN_QUERY_LENGTH",
    "ArmIdentity",
    "ArmIndexMap",
    "FeatureTransformer",
    "sample_thompson",
    "score_arms",
]
