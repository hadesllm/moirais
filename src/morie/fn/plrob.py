# morie.fn — function file (hadesllm/morie)
"""Partially linear Robinson estimator."""

from __future__ import annotations

import numpy as np
from scipy import stats


def plrob(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Robinson (1988) double-residual estimator for the partially linear model.

    Model: :math:`Y = X'\beta + g(Z) + \varepsilon`. The estimator:

    1. Regress :math:`Y` on :math:`Z` nonparametrically: :math:`\tilde{Y} = Y - E[Y|Z]`.
    2. Regress each :math:`X_j` on :math:`Z`: :math:`\tilde{X}_j = X_j - E[X_j|Z]`.
    3. OLS of :math:`\tilde{Y}` on :math:`\tilde{X}`: :math:`\hat{\beta}`.

    Parameters
    ----------
    y : np.ndarray
        Response (n,).
    X : np.ndarray
        Linear covariates (n, p).
    Z : np.ndarray
        Nonparametric covariate (n,).
    bandwidth : float or None
        Kernel bandwidth for nonparametric regressions.
    kernel : str
        Kernel function.

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``residuals``,
        ``bandwidth``, ``n_obs``.

    References
    ----------
    Robinson, P. M. (1988). Root-N-consistent semiparametric regression.
        Econometrica, 56, 931-954.
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

    from morie.fn.nwker import nwker

    kw = {"bandwidth": bandwidth, "kernel": kernel}
    ey = np.asarray(nwker(Z, y, **kw)["y_hat"])
    y_tilde = y - ey

    X_tilde = np.empty_like(X)
    for j in range(p):
        ex = np.asarray(nwker(Z, X[:, j], **kw)["y_hat"])
        X_tilde[:, j] = X[:, j] - ex

    XtX = X_tilde.T @ X_tilde
    try:
        beta = np.linalg.solve(XtX, X_tilde.T @ y_tilde)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, X_tilde.T @ y_tilde, rcond=None)[0]

    resid = y_tilde - X_tilde @ beta
    sigma2 = float(np.sum(resid**2) / (n - p))
    try:
        cov = sigma2 * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(XtX)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.t.sf(np.abs(t_stat), df=n - p)

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "residuals": resid.tolist(),
        "bandwidth": bandwidth,
        "n_obs": n,
    }


plrob_fn = plrob


def cheatsheet() -> str:
    return "plrob({y, X, Z}) -> Robinson partially linear estimator."
