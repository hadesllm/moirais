# morie.fn — function file (hadesllm/morie)
"""HC0/HC1/HC2/HC3 heteroskedasticity-robust standard errors."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def hc_robust_se(
    y: np.ndarray,
    X: np.ndarray,
    *,
    hc_type: str = "HC1",
    add_intercept: bool = True,
) -> RegressionResult:
    """OLS with HC0/HC1/HC2/HC3 heteroskedasticity-consistent standard errors.

    HC0: :math:`\\text{diag}(e_i^2)` (White, 1980)
    HC1: :math:`\\frac{n}{n-k} \\text{diag}(e_i^2)` (finite-sample correction)
    HC2: :math:`\\text{diag}(e_i^2 / (1 - h_{ii}))` (MacKinnon & White, 1985)
    HC3: :math:`\\text{diag}(e_i^2 / (1 - h_{ii})^2)` (jackknife-like)

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    hc_type : str
        One of 'HC0', 'HC1', 'HC2', 'HC3'.
    add_intercept : bool

    Returns
    -------
    RegressionResult
        With heteroskedasticity-robust standard errors.

    References
    ----------
    MacKinnon, J. G. & White, H. (1985). Some heteroskedasticity-consistent
    covariance matrix estimators with improved finite sample properties.
    *J. Econometrics*, 29(3), 305--325.
    """
    hc_type = hc_type.upper()
    if hc_type not in ("HC0", "HC1", "HC2", "HC3"):
        raise ValueError("hc_type must be HC0, HC1, HC2, or HC3.")

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

    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    e2 = resid ** 2
    if hc_type == "HC0":
        omega = e2
    elif hc_type == "HC1":
        omega = e2 * n / (n - k)
    elif hc_type == "HC2":
        omega = e2 / (1.0 - h)
    else:
        omega = e2 / (1.0 - h) ** 2

    meat = (X * omega[:, None]).T @ X
    cov_hc = XtX_inv @ meat @ XtX_inv
    se_arr = np.sqrt(np.diag(cov_hc).clip(0))

    t_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=n - k)

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method=f"OLS ({hc_type})",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        residuals=resid,
        fitted=X @ beta,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"hc_type": hc_type},
    )


hc012 = hc_robust_se


def cheatsheet() -> str:
    return "hc_robust_se({}) -> OLS with HC0/HC1/HC2/HC3 robust SE."
