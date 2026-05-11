"""Weighted least squares regression."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def wls_regression(
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    *,
    add_intercept: bool = True,
) -> RegressionResult:
    """Weighted least squares via :math:`(X^\\top W X)^{-1} X^\\top W y`.

    Parameters
    ----------
    y : (n,) array
    X : (n, p) array
    weights : (n,) positive weights (inverse variance)
    add_intercept : bool

    Returns
    -------
    RegressionResult

    References
    ----------
    Kutner, M. H. et al. (2005). *Applied Linear Statistical Models* (5th ed.).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if w.shape[0] != n or y.shape[0] != n:
        raise ValueError("y, X, weights must have the same number of rows.")
    if np.any(w <= 0):
        raise ValueError("All weights must be positive.")

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    W = np.diag(w)
    XtWX = X.T @ W @ X
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        raise ValueError("X'WX is singular.")

    beta = XtWX_inv @ (X.T @ W @ y)
    fitted = X @ beta
    resid = y - fitted
    ss_res = float(w @ (resid ** 2))
    y_bar_w = np.sum(w * y) / np.sum(w)
    ss_tot = float(w @ ((y - y_bar_w) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0

    sigma2 = ss_res / (n - k)
    cov_beta = sigma2 * XtWX_inv
    se_arr = np.sqrt(np.diag(cov_beta).clip(0))
    t_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=n - k)

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="WLS",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        adj_r_squared=adj_r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"sigma2": sigma2},
    )


wlsrg = wls_regression


def cheatsheet() -> str:
    return "wls_regression({}) -> Weighted least squares regression."
