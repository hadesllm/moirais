# morie.fn — function file (hadesllm/morie)
"""Partially-linear regression (Robinson 1988; Horowitz 2009, Ch 3).

Model:   Y = X * beta + g(Z) + e

Robinson estimator:
    1. m_Y(z) = E[Y|Z=z], m_X(z) = E[X|Z=z]    (Nadaraya-Watson)
    2. r_Y = Y - m_Y(Z), r_X = X - m_X(Z)
    3. beta_hat = (r_X' r_X)^{-1} r_X' r_Y
SE via the heteroskedasticity-consistent (HC0) sandwich on r_X.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_plr_estimator"]


def _silverman(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _nw_loo(z_train, y_train, h):
    """Leave-one-out Nadaraya-Watson smoother evaluated on the training set."""
    if z_train.ndim == 1:
        u = (z_train[:, None] - z_train[None, :]) / h
        w = np.exp(-0.5 * u * u)
    else:
        d = (z_train[:, None, :] - z_train[None, :, :]) / h
        w = np.exp(-0.5 * (d * d).sum(axis=2))
    np.fill_diagonal(w, 0.0)
    wsum = w.sum(axis=1)
    safe = np.where(wsum > 0, wsum, 1.0)
    return (w @ y_train) / safe


def horowitz_plr_estimator(x, y, z, bandwidth=None):
    """Robinson partially-linear estimator.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, k)
        Parametric regressors whose coefficients are of interest.
    y : array-like, shape (n,)
    z : array-like, shape (n,) or (n, d)
        Nonparametric covariates.
    bandwidth : float, optional
        Kernel bandwidth (Silverman default on z[:, 0]).

    Returns
    -------
    RichResult with payload keys: estimate, se, n, method.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != y.size:
        Z = Z.T
    n = y.size
    if n < 5 or X.shape[0] != n or Z.shape[0] != n:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "partially-linear (insufficient data)"})
    h = float(bandwidth) if bandwidth is not None else _silverman(Z[:, 0])
    if h <= 0:
        h = max(_silverman(Z[:, 0]), 1e-6)
    Zs = Z[:, 0] if Z.shape[1] == 1 else Z
    mY = _nw_loo(Zs, y, h)
    mX = np.column_stack([_nw_loo(Zs, X[:, j], h) for j in range(X.shape[1])])
    rY = y - mY
    rX = X - mX
    try:
        beta, *_ = np.linalg.lstsq(rX, rY, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.full(X.shape[1], np.nan)
    resid = rY - rX @ beta
    bread = np.linalg.pinv(rX.T @ rX)
    meat = rX.T @ (rX * (resid ** 2)[:, None])
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    est = beta if beta.size > 1 else float(beta[0])
    se_v = se if beta.size > 1 else float(se[0])
    return RichResult(payload={
        "estimate": est, "se": se_v, "bandwidth": h, "n": n,
        "method": "Robinson (1988) partially-linear regression",
    })


def cheatsheet():
    return "hrzp1: Robinson partially-linear regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(2)
    n = 500
    z = rng.uniform(-1, 1, n)
    x = rng.standard_normal(n)
    y = 1.5 * x + np.sin(np.pi * z) + 0.2 * rng.standard_normal(n)
    res = horowitz_plr_estimator(x, y, z)
    print(res)
    assert abs(res["estimate"] - 1.5) < 0.2
