# morie.fn — function file (hadesllm/morie)
"""Partially linear penalized estimator."""

from __future__ import annotations

import numpy as np
from scipy import stats


def plpen(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_basis: int | None = None,
    penalty: float = 1.0,
) -> dict:
    r"""
    Partially linear model with penalized sieve for g(Z).

    Model: :math:`Y = X'\beta + g(Z) + \varepsilon`.
    Estimates :math:`g` via penalized series regression (roughness penalty
    on the sieve coefficients), then recovers :math:`\beta`.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Linear covariates (n, p).
    Z : np.ndarray
        Nonparametric covariate (n,).
    n_basis : int or None
        Number of basis functions.
    penalty : float
        Roughness penalty lambda >= 0.

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``gamma``,
        ``penalty``, ``n_obs``.

    References
    ----------
    Horowitz (2009). Ch 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("y, X, Z must have same n.")
    if n < 10:
        raise ValueError("Need at least 10 observations.")
    if penalty < 0:
        raise ValueError("penalty must be >= 0.")

    if n_basis is None:
        n_basis = min(n // 5, 10)
    n_basis = max(1, n_basis)

    z_lo, z_hi = Z.min(), Z.max()
    z_range = z_hi - z_lo if z_hi > z_lo else 1.0
    z_scaled = (Z - z_lo) / z_range
    B = np.column_stack([z_scaled**k for k in range(1, n_basis + 1)])

    W = np.column_stack([X, B])
    P = np.zeros(p + n_basis)
    P[p:] = penalty
    P_mat = np.diag(P)

    A = W.T @ W + P_mat
    try:
        theta = np.linalg.solve(A, W.T @ y)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, W.T @ y, rcond=None)[0]

    beta = theta[:p]
    gamma = theta[p:]

    resid = y - W @ theta
    df = max(n - p - n_basis, 1)
    sigma2 = float(np.sum(resid**2) / df)
    try:
        cov_full = sigma2 * np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov_full = sigma2 * np.linalg.pinv(A)

    se = np.sqrt(np.maximum(np.diag(cov_full)[:p], 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.t.sf(np.abs(t_stat), df=df)

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "gamma": gamma.tolist(),
        "penalty": float(penalty),
        "n_obs": n,
    }


plpen_fn = plpen


def cheatsheet() -> str:
    return "plpen({y, X, Z}) -> Partially linear penalized sieve."
