# morie.fn — function file (hadesllm/morie)
"""OLS regression with full inference."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def ols_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
) -> RegressionResult:
    r"""Ordinary least squares with coefficients, SE, t, p, R2, adj-R2.

    Solves :math:`\\hat{\\beta} = (X^\\top X)^{-1} X^\\top y` and computes
    the usual OLS inference under homoscedasticity.

    Parameters
    ----------
    y : (n,) array
        Response variable.
    X : (n, p) array
        Predictor matrix.
    add_intercept : bool
        If True, prepend a column of ones.

    Returns
    -------
    RegressionResult

    References
    ----------
    Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    n, k = X.shape

    if n <= k:
        raise ValueError(f"Need n > k, got n={n}, k={k}.")

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        raise ValueError("X'X is singular; check for perfect multicollinearity.")

    beta = XtX_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0

    sigma2 = ss_res / (n - k)
    cov_beta = sigma2 * XtX_inv
    se_arr = np.sqrt(np.diag(cov_beta).clip(0))
    t_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=n - k)

    f_num = (ss_tot - ss_res) / (k - 1) if k > 1 else 0.0
    f_den = ss_res / (n - k)
    f_stat = f_num / f_den if f_den > 0 else 0.0
    f_pval = float(_st.f.sf(f_stat, k - 1, n - k)) if k > 1 else 1.0

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]

    return RegressionResult(
        method="OLS",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        adj_r_squared=adj_r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={
            "t_values": {nm: float(t) for nm, t in zip(names, t_vals)},
            "f_statistic": f_stat,
            "f_pvalue": f_pval,
            "sigma2": sigma2,
        },
    )


olsrg = ols_regression


def cheatsheet() -> str:
    return "ols_regression({}) -> OLS with full inference (coef, SE, t, p, R2, adj-R2)."
