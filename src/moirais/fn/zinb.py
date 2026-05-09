"""Zero-Inflated Negative Binomial (ZINB) model."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def zero_inflated_negbin(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """ZINB model via simplified EM.

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

    mu = np.full(n, np.mean(y) + 0.1)
    alpha = 1.0
    pi = np.sum(y == 0) / n * 0.5

    for _ in range(max_iter):
        r = 1 / max(alpha, 1e-6)
        nb_0 = np.exp(sp_stats.nbinom.logpmf(0, n=r, p=r / (r + mu)))
        tau = np.where(y == 0, pi / (pi + (1 - pi) * nb_0 + 1e-12), 0.0)

        pi_new = tau.mean()
        w = 1 - tau

        wn = w.sum() + 1e-12
        mu_new = np.sum(w * y) / wn
        mu = np.full(n, max(mu_new, 1e-6))

        var_y = np.sum(w * (y - mu) ** 2) / wn
        alpha_new = max((var_y - mu[0]) / (mu[0] ** 2 + 1e-12), 1e-6)

        if abs(pi_new - pi) < tol and abs(alpha_new - alpha) < tol:
            pi = pi_new
            alpha = alpha_new
            break
        pi = pi_new
        alpha = alpha_new

    r = 1 / max(alpha, 1e-6)
    ll = np.sum(
        np.where(
            y == 0,
            np.log(pi + (1 - pi) * np.exp(sp_stats.nbinom.logpmf(0, n=r, p=r / (r + mu))) + 1e-300),
            np.log(1 - pi + 1e-300) + sp_stats.nbinom.logpmf(y.astype(int), n=r, p=r / (r + mu)),
        )
    )

    return DescriptiveResult(
        name="zinb",
        value=float(-2 * ll),
        extra={
            "mu": float(mu[0]),
            "alpha": float(alpha),
            "zero_prob": float(pi),
            "log_likelihood": float(ll),
            "n": n,
            "n_zeros": int(np.sum(y == 0)),
        },
    )


zinb = zero_inflated_negbin


def cheatsheet() -> str:
    return "zero_inflated_negbin({}) -> Zero-Inflated Negative Binomial (ZINB) model."
