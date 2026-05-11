# morie.fn — function file (hadesllm/morie)
"""Censored regression (semiparametric)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def cnsrd(
    y: np.ndarray,
    X: np.ndarray,
    *,
    censoring: np.ndarray | None = None,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Semiparametric censored regression (Powell 1984/1986).

    Estimates :math:`\beta` in the censored model
    :math:`Y = \max(0, X'\beta + \varepsilon)` using the
    censored least absolute deviations (CLAD) approach,
    augmented with kernel smoothing for the censoring probability.

    Parameters
    ----------
    y : np.ndarray
        Response (n,), with censoring at 0.
    X : np.ndarray
        Covariates (n, p).
    censoring : np.ndarray or None
        Censoring indicator (n,): 1 = observed, 0 = censored.
        If None, inferred as y > 0.
    bandwidth : float or None
        Kernel bandwidth for conditional censoring estimation.
    kernel : str
        Kernel function.

    Returns
    -------
    dict
        ``beta``, ``se``, ``t_stat``, ``pval``, ``n_censored``,
        ``n_obs``.

    References
    ----------
    Powell, J. L. (1984). Least absolute deviations estimation for the
        censored regression model. JoE, 25, 303-325.
    Horowitz (2009). Ch 7.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same n.")
    if n < p + 5:
        raise ValueError(f"Need at least {p + 5} observations.")

    if censoring is None:
        censoring = (y > 0).astype(float)
    else:
        censoring = np.asarray(censoring, dtype=float).ravel()

    n_cens = int((censoring == 0).sum())

    def clad_obj(b):
        xb = X @ b
        resid = y - np.maximum(0, xb)
        return float(np.mean(np.abs(resid)))

    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    res = minimize(clad_obj, b0, method="L-BFGS-B",
                   options={"maxiter": 200})
    beta = res.x

    xb = X @ beta
    resid = y - np.maximum(0, xb)
    uncensored = censoring == 1

    if uncensored.sum() > p:
        sigma2 = float(np.sum(resid[uncensored]**2) / max(uncensored.sum() - p, 1))
    else:
        sigma2 = float(np.mean(resid**2))

    from morie.fn.nwker import _get_kernel, _silverman_bw

    k_fn = _get_kernel(kernel)
    h = bandwidth if bandwidth is not None else _silverman_bw(xb)

    diff = xb[:, None] - xb[None, :]
    K = k_fn(diff / h)
    denom = K.sum(axis=1)
    denom = np.where(denom < 1e-15, 1.0, denom)
    p_uncens = (K @ censoring) / denom
    p_uncens = np.maximum(p_uncens, 0.01)

    W = np.diag(p_uncens)
    XtWX = X.T @ W @ X
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(XtWX)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stat = beta / np.where(se > 0, se, np.inf)
    pval = 2 * stats.norm.sf(np.abs(t_stat))

    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stat": t_stat.tolist(),
        "pval": pval.tolist(),
        "n_censored": n_cens,
        "n_obs": n,
    }


cnsrd_fn = cnsrd


def cheatsheet() -> str:
    return "cnsrd({y, X}) -> Semiparametric censored regression (CLAD)."
