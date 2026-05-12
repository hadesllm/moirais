# morie.fn -- function file (hadesllm/morie)
"""Poisson regression via IRLS."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def poisson_regression(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> DescriptiveResult:
    """Poisson GLM via iteratively reweighted least squares.

    Parameters
    ----------
    y_counts : (n,) non-negative integer counts
    X : (n, p) predictors

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(y_counts, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X_int @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        W = np.diag(mu + 1e-8)
        z = eta + (y - mu) / (mu + 1e-8)
        try:
            beta_new = np.linalg.solve(X_int.T @ W @ X_int, X_int.T @ W @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_final = np.exp(np.clip(X_int @ beta, -20, 20))
    deviance = 2 * np.sum(y * np.log((y + 1e-12) / (mu_final + 1e-12)) - (y - mu_final))
    aic = deviance + 2 * k

    try:
        cov_beta = np.linalg.inv(X_int.T @ np.diag(mu_final) @ X_int)
        se = np.sqrt(np.diag(cov_beta))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    z_vals = beta / (se + 1e-12)
    p_vals = 2 * sp_stats.norm.sf(np.abs(z_vals))

    coef_names = ["intercept"] + [f"x{j}" for j in range(p)]

    return DescriptiveResult(
        name="poisson",
        value=float(deviance),
        extra={
            "coefficients": dict(zip(coef_names, beta.tolist())),
            "se": dict(zip(coef_names, se.tolist())),
            "p_values": dict(zip(coef_names, p_vals.tolist())),
            "deviance": float(deviance),
            "aic": float(aic),
            "n": n,
            "k": k,
        },
    )


pois = poisson_regression


def cheatsheet() -> str:
    return "poisson_regression({}) -> Poisson regression via IRLS."
