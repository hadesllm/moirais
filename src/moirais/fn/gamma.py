# moirais.fn — function file (hadesllm/moirais)
"""Gamma GLM with log link."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def gamma_glm(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> RegressionResult:
    """Gamma GLM with log link via IRLS.

    Variance function :math:`V(\\mu) = \\mu^2` (constant CV model).

    Parameters
    ----------
    y : (n,) positive continuous responses
    X : (n, p) predictors
    add_intercept : bool
    max_iter : int
    tol : float

    Returns
    -------
    RegressionResult

    References
    ----------
    McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.any(y <= 0):
        raise ValueError("Gamma GLM requires strictly positive y.")
    n, p_raw = X.shape
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
    k = X.shape[1]

    beta = np.zeros(k)
    beta[0] = np.log(np.mean(y))

    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        W = mu ** 2
        z = eta + (y - mu) / (mu + 1e-12)
        XtWX = (X * W[:, None]).T @ X
        XtWz = (X * W[:, None]).T @ z
        try:
            beta_new = np.linalg.solve(XtWX + np.eye(k) * 1e-10, XtWz)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_f = np.exp(np.clip(X @ beta, -20, 20))
    resid = y - mu_f
    deviance = 2.0 * float(np.sum(-np.log(y / (mu_f + 1e-300)) + (y - mu_f) / (mu_f + 1e-300)))
    phi = deviance / (n - k)
    aic = deviance / phi + 2 * k

    W_f = mu_f ** 2
    XtWX = (X * W_f[:, None]).T @ X
    try:
        cov = phi * np.linalg.inv(XtWX)
        se_arr = np.sqrt(np.diag(cov).clip(0))
    except np.linalg.LinAlgError:
        se_arr = np.full(k, float("nan"))

    z_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.norm.sf(np.abs(z_vals))

    names = (["(Intercept)"] if add_intercept else []) + [
        f"x{j}" for j in range(p_raw)
    ]
    return RegressionResult(
        method="Gamma GLM (log link)",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        fitted=mu_f,
        residuals=resid,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"deviance": deviance, "phi": phi, "aic": aic},
    )


gamma_glm_fn = gamma_glm


def cheatsheet() -> str:
    return "gamma_glm({}) -> Gamma GLM with log link via IRLS."
