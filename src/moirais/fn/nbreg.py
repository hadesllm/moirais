# moirais.fn — function file (hadesllm/moirais)
"""Negative binomial regression."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

from ._containers import DescriptiveResult


def negbin_regression(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> DescriptiveResult:
    """NB2 regression via IRLS with MLE for dispersion.

    Parameters
    ----------
    y_counts : (n,) counts
    X : (n, p)

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(y_counts, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    beta = np.zeros(k)
    alpha = 1.0

    for _ in range(max_iter):
        eta = X_int @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        W = mu / (1 + alpha * mu + 1e-8)
        z = eta + (y - mu) / (mu + 1e-8)
        try:
            beta_new = np.linalg.solve(X_int.T @ (X_int * W[:, None]), X_int.T @ (W * z))
        except np.linalg.LinAlgError:
            break

        mu_new = np.exp(np.clip(X_int @ beta_new, -20, 20))

        def neg_ll(a):
            a = max(a, 1e-6)
            ll = np.sum(sp_stats.nbinom.logpmf(y.astype(int), n=1 / a, p=1 / (1 + a * mu_new)))
            return -ll

        res = minimize_scalar(neg_ll, bounds=(1e-6, 100), method="bounded")
        alpha = res.x

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu_final = np.exp(np.clip(X_int @ beta, -20, 20))
    ll = float(np.sum(sp_stats.nbinom.logpmf(y.astype(int), n=1 / alpha, p=1 / (1 + alpha * mu_final))))

    coef_names = ["intercept"] + [f"x{j}" for j in range(p)]

    return DescriptiveResult(
        name="negbin",
        value=float(-2 * ll),
        extra={
            "coefficients": dict(zip(coef_names, beta.tolist())),
            "alpha": float(alpha),
            "log_likelihood": float(ll),
            "aic": float(-2 * ll + 2 * (k + 1)),
            "n": n,
            "k": k,
        },
    )


nbreg = negbin_regression


def cheatsheet() -> str:
    return "negbin_regression({}) -> Negative binomial regression."
