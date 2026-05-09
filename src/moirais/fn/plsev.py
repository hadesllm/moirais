# moirais.fn — function file (hadesllm/moirais)
"""Partially linear sieve estimator."""

from __future__ import annotations

import numpy as np
from scipy import stats


def plsev(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_basis: int | None = None,
    basis: str = "polynomial",
) -> dict:
    r"""
    Partially linear model via sieve (series) estimation.

    Model: :math:`Y = X'\beta + g(Z) + \varepsilon`.
    Approximates :math:`g(Z) \approx \sum_{k=1}^{K} \gamma_k \phi_k(Z)`
    using a series basis, then jointly estimates :math:`\beta, \gamma`
    by OLS.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Linear covariates (n, p).
    Z : np.ndarray
        Nonparametric covariate (n,).
    n_basis : int or None
        Number of basis functions K. Default min(n//5, 10).
    basis : str
        ``'polynomial'`` (power series) or ``'cosine'`` (Fourier cosine).

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``gamma``
        (basis coefficients), ``n_basis``, ``n_obs``.

    References
    ----------
    Horowitz (2009). Ch 3, eq. 3.17.
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

    if n_basis is None:
        n_basis = min(n // 5, 10)
    n_basis = max(1, n_basis)

    if basis not in ("polynomial", "cosine"):
        raise ValueError("basis must be 'polynomial' or 'cosine'.")

    z_lo, z_hi = Z.min(), Z.max()
    z_range = z_hi - z_lo if z_hi > z_lo else 1.0
    z_scaled = (Z - z_lo) / z_range

    if basis == "polynomial":
        B = np.column_stack([z_scaled**k for k in range(1, n_basis + 1)])
    else:
        B = np.column_stack([np.cos(np.pi * k * z_scaled) for k in range(1, n_basis + 1)])

    W = np.column_stack([X, B])
    WtW = W.T @ W
    try:
        theta = np.linalg.solve(WtW, W.T @ y)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(WtW, W.T @ y, rcond=None)[0]

    beta = theta[:p]
    gamma = theta[p:]

    resid = y - W @ theta
    df = n - p - n_basis
    sigma2 = float(np.sum(resid**2) / max(df, 1))
    try:
        cov = sigma2 * np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(WtW)

    se = np.sqrt(np.maximum(np.diag(cov)[:p], 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.t.sf(np.abs(t_stat), df=max(df, 1))

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "gamma": gamma.tolist(),
        "n_basis": n_basis,
        "n_obs": n,
    }


plsev_fn = plsev


def cheatsheet() -> str:
    return "plsev({y, X, Z}) -> Partially linear sieve estimator."
