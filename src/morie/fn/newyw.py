# morie.fn -- function file (hadesllm/morie)
"""Newey-West HAC standard errors."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def newey_west(
    y: np.ndarray,
    X: np.ndarray,
    *,
    max_lag: int | None = None,
    add_intercept: bool = True,
) -> RegressionResult:
    r"""OLS with Newey-West heteroskedasticity and autocorrelation consistent SE.

    Uses the Bartlett kernel with automatic lag selection
    :math:`\\lfloor 4(n/100)^{2/9} \\rfloor` if *max_lag* is not given.

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    max_lag : int or None
        Maximum lag for HAC. Auto-selected if None.
    add_intercept : bool

    Returns
    -------
    RegressionResult
        With HAC (Newey-West) standard errors.

    References
    ----------
    Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    *Econometrica*, 55(3), 703--708.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lag = max(max_lag, 0)

    S = np.zeros((k, k))
    for t in range(n):
        S += resid[t] ** 2 * np.outer(X[t], X[t])

    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1)
        for t in range(lag, n):
            cross = resid[t] * resid[t - lag] * np.outer(X[t], X[t - lag])
            S += w * (cross + cross.T)

    cov_nw = XtX_inv @ S @ XtX_inv
    se_arr = np.sqrt(np.diag(cov_nw).clip(0))
    t_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=n - k)

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="OLS (Newey-West HAC)",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        residuals=resid,
        fitted=X @ beta,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"max_lag": max_lag},
    )


newyw = newey_west


def cheatsheet() -> str:
    return "newey_west({}) -> OLS with Newey-West HAC standard errors."
