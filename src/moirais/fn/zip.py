"""Zero-Inflated Poisson (ZIP) model."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def zero_inflated_poisson(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """ZIP model via EM algorithm.

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

    pi = 0.3
    lam = np.mean(y) + 0.1
    beta = np.zeros(X_int.shape[1])
    beta[0] = np.log(lam)

    for _ in range(max_iter):
        mu = np.exp(np.clip(X_int @ beta, -20, 20))
        pois_0 = np.exp(-mu)
        tau = np.where(y == 0, pi / (pi + (1 - pi) * pois_0 + 1e-12), 0.0)

        w = 1 - tau
        w_sum = w.sum() + 1e-12
        pi_new = tau.sum() / n

        for _irls in range(10):
            eta = X_int @ beta
            mu = np.exp(np.clip(eta, -20, 20))
            W = w * mu
            z = eta + (y - mu) / (mu + 1e-8)
            try:
                beta_new = np.linalg.solve(X_int.T @ (X_int * W[:, None]), X_int.T @ (W * z))
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        if abs(pi_new - pi) < tol:
            pi = pi_new
            break
        pi = pi_new

    mu_final = np.exp(np.clip(X_int @ beta, -20, 20))
    ll = np.sum(
        np.where(
            y == 0,
            np.log(pi + (1 - pi) * np.exp(-mu_final) + 1e-300),
            np.log(1 - pi + 1e-300) + sp_stats.poisson.logpmf(y.astype(int), mu_final),
        )
    )

    coef_names = ["intercept"] + [f"x{j}" for j in range(p)]

    return DescriptiveResult(
        name="zip",
        value=float(-2 * ll),
        extra={
            "coefficients": dict(zip(coef_names, beta.tolist())),
            "zero_prob": float(pi),
            "log_likelihood": float(ll),
            "n": n,
            "n_zeros": int(np.sum(y == 0)),
        },
    )


zip_ = zero_inflated_poisson


def cheatsheet() -> str:
    return "zero_inflated_poisson({}) -> Zero-Inflated Poisson (ZIP) model."
