# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian ridge regression."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy import stats


def bayesian_ridge(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    alpha: float = 1.0,
    beta_noise: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Bayesian ridge regression with evidence maximization for hyperparameters.

    Prior: beta ~ N(0, alpha^{-1} I)
    Likelihood: y | X, beta ~ N(X beta, beta_noise^{-1} I)

    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param alpha: Initial prior precision.
    :param beta_noise: Initial noise precision.
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :param prob: Credible interval probability.
    :return: Dictionary with posterior mean, alpha, beta_noise, log_evidence.

    References
    ----------
    MacKay, D. J. C. (1992). *Neural Computation*, 4(3), 415--447.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    eigenvalues = np.linalg.eigvalsh(X_arr.T @ X_arr)

    for _ in range(max_iter):
        S_inv = alpha * np.eye(p) + beta_noise * X_arr.T @ X_arr
        S = np.linalg.inv(S_inv)
        m = beta_noise * S @ X_arr.T @ y_arr

        gamma = float(np.sum(beta_noise * eigenvalues / (alpha + beta_noise * eigenvalues)))
        alpha_new = gamma / float(m @ m + 1e-30)
        resid = y_arr - X_arr @ m
        beta_new = (n - gamma) / float(resid @ resid + 1e-30)

        if abs(alpha_new - alpha) < tol and abs(beta_new - beta_noise) < tol:
            alpha = alpha_new
            beta_noise = beta_new
            break
        alpha = alpha_new
        beta_noise = beta_new

    S_inv = alpha * np.eye(p) + beta_noise * X_arr.T @ X_arr
    S = np.linalg.inv(S_inv)
    m = beta_noise * S @ X_arr.T @ y_arr

    se = np.sqrt(np.diag(S))
    z = stats.norm.ppf(1 - (1 - prob) / 2)

    return {
        "posterior_mean": m.tolist(),
        "posterior_cov": S.tolist(),
        "alpha": float(alpha),
        "beta_noise": float(beta_noise),
        "ci_lower": (m - z * se).tolist(),
        "ci_upper": (m + z * se).tolist(),
        "sigma2": float(1.0 / beta_noise),
    }


brdge = bayesian_ridge


def cheatsheet() -> str:
    return "bayesian_ridge({}) -> Bayesian ridge regression."
