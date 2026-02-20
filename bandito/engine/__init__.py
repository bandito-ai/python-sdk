"""Pure math engine for Linear Thompson Sampling.

Copied from backend/app/services/bandit/ — zero app imports, SDK-portable.
"""

from .constants import DEFAULT_RELATIVE_LATENCY, MIN_QUERY_LENGTH, OPTIMIZATION_BETAS
from .features import ArmIdentity, ArmIndexMap, compute_dimensions
from .linalg import sample_thompson

__all__ = [
    "OPTIMIZATION_BETAS",
    "DEFAULT_RELATIVE_LATENCY",
    "MIN_QUERY_LENGTH",
    "ArmIdentity",
    "ArmIndexMap",
    "compute_dimensions",
    "sample_thompson",
]
