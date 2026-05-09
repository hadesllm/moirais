# moirais.fn — function file (hadesllm/moirais)
"""Probit regression via IRLS."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import RegressionResult


def probit_regression(y: np.ndarray, X: np.ndarray, cdf=None, *, add_intercept: bool = True, max_iter: int = 50, tol: float = 1e-8) -> RegressionResult:
    """Probit regression via IRLS (Fisher scoring).

    The link function is :math:`\\Phi^{-1}(\\mu)` where :math:`\\Phi`
    is the standard normal CDF.

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

    References
    ----------
    Bliss, C. I. (1934). The method of probits. *Science*, 79(2037), 38--39.
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
        mu = _st.norm.cdf(eta)
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        phi = _st.norm.pdf(eta)
        W = (phi ** 2) / (mu * (1.0 - mu)) + 1e-12
        z = eta + (y - mu) / (phi + 1e-12)
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

    eta_f = X @ beta
    mu_f = _st.norm.cdf(eta_f)
    mu_f = np.clip(mu_f, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(mu_f) + (1 - y) * np.log(1 - mu_f)))
    deviance = -2.0 * ll
    aic = deviance + 2 * k

    phi_f = _st.norm.pdf(eta_f)
    W_f = (phi_f ** 2) / (mu_f * (1 - mu_f)) + 1e-12
    XtWX = (X * W_f[:, None]).T @ X
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
        method="Probit (IRLS)",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        fitted=mu_f,
        residuals=y - mu_f,
        n=n,
        k=k - (1 if add_intercept else 0),
        extra={"deviance": deviance, "aic": aic, "log_likelihood": ll},
    )


probt = probit_regression


def cheatsheet() -> str:
    return "probit_regression({}) -> Probit regression via IRLS."
