# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Semiparametric Box-Cox transformation model."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def bxcsm(
    y: np.ndarray,
    X: np.ndarray,
    *,
    lambda_range: tuple[float, float] = (-2.0, 2.0),
) -> dict:
    r"""
    Semiparametric Box-Cox transformation model.

    Estimates the transformation parameter :math:`\lambda` in:

    .. math::

        T_\lambda(Y) = X'\beta + \varepsilon

    where :math:`T_\lambda(y) = (y^\lambda - 1)/\lambda` for
    :math:`\lambda \neq 0` and :math:`\log y` for :math:`\lambda = 0`.
    Uses profile likelihood over :math:`\lambda`.

    Parameters
    ----------
    y : np.ndarray
        Response (n,), must be positive.
    X : np.ndarray
        Covariates (n, p).
    lambda_range : tuple
        Search interval for lambda.

    Returns
    -------
    dict
        ``lambda_opt``, ``beta``, ``se``, ``t_stat``, ``pval``,
        ``log_likelihood``, ``n_obs``.

    References
    ----------
    Box, G. & Cox, D. (1964). An analysis of transformations. JRSS-B.
    Horowitz (2009). Ch 6.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same n.")
    if np.any(y <= 0):
        raise ValueError("y must be strictly positive for Box-Cox.")
    if n < p + 2:
        raise ValueError(f"Need at least {p + 2} observations.")

    def _transform(y_arr, lam):
        if abs(lam) < 1e-10:
            return np.log(y_arr)
        return (y_arr**lam - 1) / lam

    def neg_profile_ll(lam):
        yt = _transform(y, lam)
        Xd = np.column_stack([np.ones(n), X])
        try:
            beta = np.linalg.lstsq(Xd, yt, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 1e15
        resid = yt - Xd @ beta
        sigma2 = np.sum(resid**2) / n
        if sigma2 <= 0:
            return 1e15
        ll = -n / 2 * np.log(sigma2) + (lam - 1) * np.sum(np.log(y))
        return -ll

    res = minimize_scalar(neg_profile_ll, bounds=lambda_range, method="bounded")
    lam_opt = float(res.x)

    yt = _transform(y, lam_opt)
    Xd = np.column_stack([np.ones(n), X])
    beta_full = np.linalg.lstsq(Xd, yt, rcond=None)[0]
    beta = beta_full[1:]
    intercept = beta_full[0]

    resid = yt - Xd @ beta_full
    sigma2 = float(np.sum(resid**2) / max(n - p - 1, 1))
    try:
        cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(Xd.T @ Xd)

    se = np.sqrt(np.maximum(np.diag(cov)[1:], 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.t.sf(np.abs(t_stat), df=max(n - p - 1, 1))

    ll = -float(res.fun)

    return {
        "lambda_opt": lam_opt,
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "intercept": float(intercept),
        "log_likelihood": ll,
        "n_obs": n,
    }


bxcsm_fn = bxcsm


def cheatsheet() -> str:
    return "bxcsm({y, X}) -> Semiparametric Box-Cox transformation model."
