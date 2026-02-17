"""
Single source of truth for bandit engine constants.

Pure module — zero imports. Safe for SDK and server.
"""

# Thompson Sampling exploration parameter (beta) per optimization mode.
# Higher beta = more exploration (wider sampling from posterior).
# "explore" favors discovering arm quality, "maximize" exploits known-best.
OPTIMIZATION_BETAS: dict[str, float] = {
    "explore": 1.5,
    "base": 1.0,
    "maximize": 0.5,
}

# Normalization ceilings for composite reward penalties.
# Cost in dollars, latency in milliseconds.
MAX_COST: float = 5.0
MAX_LATENCY: float = 60_000.0

# Jitter added to A_inv diagonal before Cholesky decomposition.
# Prevents numerical failure when A is near-singular.
CHOLESKY_JITTER: float = 1e-6

# Cold-start defaults for context features when values are unknown.
# log(1) = 0.0 is neutral for query-length interaction terms.
MIN_QUERY_LENGTH: int = 1
# 1.0 means "average latency" — no interaction effect.
DEFAULT_RELATIVE_LATENCY: float = 1.0
