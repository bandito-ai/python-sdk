"""
Bayesian linear algebra for shared Linear Thompson Sampling.

Pure module — imports only numpy and constants. Safe for SDK and server.

Mathematical core:
    - Bayesian linear regression: maintains precision matrix A and accumulator b
    - Posterior: theta = A^{-1} b, covariance = A^{-1}
    - Thompson Sampling: sample theta_tilde ~ N(theta, beta^2 * A^{-1})
    - Arm scoring: score_a = x_a^T * theta_tilde

All functions operate on raw numpy arrays. No ORM, no async, no DB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import CHOLESKY_JITTER


@dataclass
class PosteriorState:
    """In-memory representation of a bandit's Bayesian state.

    Attributes:
        A: Precision matrix (d x d). Starts as identity, grows with observations.
        b: Reward accumulator (d,). Sum of x * reward over all observations.
        theta: Posterior mean weights (d,). theta = A^{-1} b.
        chol: Cholesky factor of A^{-1} (d x d). L such that L L^T = A^{-1} + jitter*I.
        dimensions: Feature vector dimensionality.
    """
    A: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    chol: np.ndarray
    dimensions: int


def bayesian_update_full(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, reward: float
) -> tuple[np.ndarray, np.ndarray]:
    """Full Bayesian update: incorporate a new observation.

    Updates both the precision matrix and reward accumulator:
        A_new = A + x x^T
        b_new = b + x * reward

    Used for first-time observations (immediate reward or first human reward).

    Args:
        A: Current precision matrix (d x d).
        b: Current reward accumulator (d,).
        x: Feature vector for the arm (d,).
        reward: Computed composite reward scalar.

    Returns:
        Tuple of (A_new, b_new). Caller must call compute_posterior() after.
    """
    A_new = A + np.outer(x, x)
    b_new = b + x * reward
    return A_new, b_new


def bayesian_update_delta(
    b: np.ndarray, x: np.ndarray, new_reward: float, old_reward: float
) -> np.ndarray:
    """Delta update to reward accumulator (A unchanged).

    Corrects a previous reward without touching the precision matrix:
        b_new = b + x * (new_reward - old_reward)

    Used when a human reward replaces or updates an existing reward.
    A is unchanged because x x^T was already added in the original update.

    Mathematical justification:
        Original: b_old = ... + x * r_old
        We want:  b_new = ... + x * r_new
        Delta:    b_new = b_old + x * (r_new - r_old)

    Args:
        b: Current reward accumulator (d,).
        x: Feature vector for the arm (d,).
        new_reward: New composite reward value.
        old_reward: Previous composite reward being replaced.

    Returns:
        Updated b_new. A is unchanged — caller uses existing A.
    """
    return b + x * (new_reward - old_reward)


def compute_posterior(
    A: np.ndarray, b: np.ndarray, jitter: float = CHOLESKY_JITTER
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior mean and Cholesky factor from precision matrix.

    Solves:
        theta = A^{-1} b
        L = cholesky(A^{-1} + jitter * I)

    Uses numpy.linalg.solve instead of explicit inversion for numerical stability.

    Args:
        A: Precision matrix (d x d). Must be positive definite.
        b: Reward accumulator (d,).
        jitter: Diagonal jitter for Cholesky stability.

    Returns:
        Tuple of (theta, chol) where theta is posterior mean and chol is
        the lower-triangular Cholesky factor of the (jittered) covariance.
    """
    theta = np.linalg.solve(A, b)
    A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
    chol = safe_cholesky(A_inv, jitter)
    return theta, chol


def safe_cholesky(
    M: np.ndarray, jitter: float = CHOLESKY_JITTER
) -> np.ndarray:
    """Cholesky decomposition with static diagonal jitter.

    Computes cholesky(M + jitter * I). The small jitter (1e-6 default)
    prevents numerical failure from floating-point imprecision without
    meaningfully affecting the posterior.

    With our feature design (one-hot + interactions) and identity prior,
    the precision matrix A stays well-conditioned. If Cholesky fails even
    with jitter, something is fundamentally wrong and we should fail loudly.

    Args:
        M: Symmetric matrix to decompose (d x d).
        jitter: Diagonal jitter added for numerical stability.

    Returns:
        Lower-triangular Cholesky factor L such that L L^T ≈ M.

    Raises:
        np.linalg.LinAlgError: If decomposition fails.
    """
    d = M.shape[0]
    return np.linalg.cholesky(M + jitter * np.eye(d))


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


def score_arms(
    theta_tilde: np.ndarray, feature_vecs: list[np.ndarray]
) -> np.ndarray:
    """Score each arm using a sampled weight vector.

    Computes:
        score_a = x_a^T * theta_tilde

    for each arm's feature vector x_a.

    Args:
        theta_tilde: Sampled weight vector from sample_thompson() (d,).
        feature_vecs: List of feature vectors, one per arm (each shape (d,)).

    Returns:
        Array of scores, one per arm. Highest score = recommended arm.
    """
    X = np.array(feature_vecs)
    return X @ theta_tilde
