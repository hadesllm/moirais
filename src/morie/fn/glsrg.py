# morie.fn -- function file (hadesllm/morie)
"""Generalized least squares regression."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def gls_regression(
    y: np.ndarray,
    X: np.ndarray,
    Omega: np.ndarray,
    *,
    add_intercept: bool = True,
) -> RegressionResult:
    r"""GLS: :math:`\\hat{\\beta} = (X^\\top \\Omega^{-1} X)^{-1} X^\\top \\Omega^{-1} y`.

    Parameters
    ----------
    y : (n,) array
    X : (n, p) array
    Omega : (n, n) positive-definite error covariance matrix
    add_intercept : bool

    Returns
    -------
    RegressionResult

    References
    ----------
    Aitken, A. C. (1935). On least squares and linear combination of
    observations. *Proc. R. Soc. Edinburgh*, 55, 42--48.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    try:
        L = np.linalg.cholesky(Omega)
    except np.linalg.LinAlgError:
        raise ValueError("Omega must be positive definite.")

    L_inv = np.linalg.inv(L)
    Omega_inv = L_inv.T @ L_inv

    XtOiX = X.T @ Omega_inv @ X
    XtOiX_inv = np.linalg.inv(XtOiX)
    beta = XtOiX_inv @ (X.T @ Omega_inv @ y)
    fitted = X @ beta
    resid = y - fitted

    sigma2 = float(resid @ Omega_inv @ resid) / (n - k)
    cov_beta = sigma2 * XtOiX_inv
    se_arr = np.sqrt(np.diag(cov_beta).clip(0))
    t_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=n - k)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="GLS",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        residuals=resid,
        fitted=fitted,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"sigma2": sigma2},
    )


glsrg = gls_regression


def cheatsheet() -> str:
    return "gls_regression({}) -> Generalized least squares regression."
