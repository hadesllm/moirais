# morie.fn -- function file (hadesllm/morie)
"""Clustered standard errors for OLS."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def clustered_se(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    *,
    add_intercept: bool = True,
) -> RegressionResult:
    r"""OLS with cluster-robust (Liang-Zeger) standard errors.

    Computes the sandwich estimator:
    :math:`\\widehat{V}_{CL} = (X^\\top X)^{-1}
    \\left(\\sum_g X_g^\\top \\hat{e}_g \\hat{e}_g^\\top X_g\\right)
    (X^\\top X)^{-1} \\cdot \\frac{G}{G-1}\\frac{n-1}{n-k}`

    Parameters
    ----------
    y : (n,) response
    X : (n, p) predictors
    clusters : (n,) cluster identifiers
    add_intercept : bool

    Returns
    -------
    RegressionResult
        With cluster-robust standard errors.

    References
    ----------
    Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to
    cluster-robust inference. *JHR*, 50(2), 317--372.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    clusters = np.asarray(clusters).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    unique_cl = np.unique(clusters)
    G = len(unique_cl)

    meat = np.zeros((k, k))
    for cl in unique_cl:
        idx = clusters == cl
        Xg = X[idx]
        eg = resid[idx]
        Sg = Xg.T @ eg
        meat += np.outer(Sg, Sg)

    scale = (G / (G - 1)) * ((n - 1) / (n - k))
    cov_cl = scale * XtX_inv @ meat @ XtX_inv
    se_arr = np.sqrt(np.diag(cov_cl).clip(0))

    t_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.t.sf(np.abs(t_vals), df=G - 1)

    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="OLS (Clustered SE)",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=r2,
        residuals=resid,
        fitted=X @ beta,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"n_clusters": G},
    )


clse = clustered_se


def cheatsheet() -> str:
    return "clustered_se({}) -> OLS with cluster-robust standard errors."
