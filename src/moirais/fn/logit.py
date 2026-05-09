# moirais.fn — function file (hadesllm/moirais)
"""Logistic regression via IRLS."""

from __future__ import annotations

import numpy as np
from scipy import special
from scipy import stats as _st

from ._containers import RegressionResult


def logistic_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> RegressionResult:
    """Logistic regression via iteratively reweighted least squares (IRLS).

    Maximises the binomial log-likelihood using Fisher scoring.

    Parameters
    ----------
    y : (n,) binary {0, 1}
    X : (n, p) predictors
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult
        Coefficients are log-odds ratios.

    References
    ----------
    McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        mu = special.expit(eta)
        W = mu * (1.0 - mu) + 1e-12
        z = eta + (y - mu) / W
        XtWX = (X * W[:, None]).T @ X
        XtWz = (X * W[:, None]).T @ z
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_final = special.expit(X @ beta)
    ll = float(np.sum(y * np.log(mu_final + 1e-300) + (1 - y) * np.log(1 - mu_final + 1e-300)))
    ll_null = float(n * (np.mean(y) * np.log(np.mean(y) + 1e-300) + (1 - np.mean(y)) * np.log(1 - np.mean(y) + 1e-300)))
    deviance = -2.0 * ll
    null_deviance = -2.0 * ll_null
    aic = deviance + 2 * k
    pseudo_r2 = 1.0 - ll / ll_null if ll_null != 0 else 0.0

    W_final = mu_final * (1.0 - mu_final) + 1e-12
    XtWX = (X * W_final[:, None]).T @ X
    try:
        cov = np.linalg.inv(XtWX)
        se_arr = np.sqrt(np.diag(cov).clip(0))
    except np.linalg.LinAlgError:
        se_arr = np.full(k, float("nan"))

    z_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.norm.sf(np.abs(z_vals))

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="Logistic (IRLS)",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        r_squared=pseudo_r2,
        fitted=mu_final,
        residuals=y - mu_final,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={
            "deviance": deviance,
            "null_deviance": null_deviance,
            "aic": aic,
            "log_likelihood": ll,
        },
    )


logit = logistic_regression


def cheatsheet() -> str:
    return "logistic_regression({}) -> Logistic regression via IRLS."
