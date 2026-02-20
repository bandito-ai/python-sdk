"""
Bayesian linear algebra for shared Linear Thompson Sampling.

Pure module — imports only numpy and constants. Safe for SDK and server.

Mathematical core:
    - Thompson Sampling: sample theta_tilde ~ N(theta, beta^2 * A^{-1})
    - Arm scoring: score_a = x_a^T * theta_tilde

All functions operate on raw numpy arrays. No ORM, no async, no DB.
"""

from __future__ import annotations

import numpy as np


def sample_thompson(
    theta: np.ndarray,
    chol: np.ndarray,
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw one sample from the Thompson Sampling posterior.

    Samples:
        theta_tilde = theta + beta * L * epsilon
    where epsilon ~ N(0, I) and L is the Cholesky factor of the covariance.

    beta controls exploration:
        - beta > 1: more exploration (wider samples)
        - beta = 1: standard Thompson Sampling
        - beta < 1: more exploitation (samples closer to mean)

    Args:
        theta: Posterior mean (d,).
        chol: Lower-triangular Cholesky factor of covariance (d x d).
        beta: Exploration scaling factor (from OPTIMIZATION_BETAS).
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        Sampled weight vector theta_tilde (d,).

    Example:
        >>> rng = np.random.default_rng(42)
        >>> theta_tilde = sample_thompson(theta, chol, beta=1.0, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()
    epsilon = rng.standard_normal(theta.shape[0])
    return theta + beta * chol @ epsilon
