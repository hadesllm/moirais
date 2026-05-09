# moirais.fn — function file (hadesllm/moirais)
"""Latent profile analysis (GMM for continuous data)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def latent_profile(
    X: np.ndarray,
    n_profiles: int = 2,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int | None = None,
) -> DescriptiveResult:
    """Gaussian mixture model (diagonal covariance) for LPA.

    Parameters
    ----------
    X : (n, p) array
    n_profiles : int
    max_iter, tol : convergence
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    K = n_profiles
    rng = np.random.default_rng(seed)

    pi = np.ones(K) / K
    idx = rng.choice(n, K, replace=False)
    mu = X[idx].copy()
    sigma2 = np.full((K, p), np.var(X, axis=0) + 1e-6)

    ll_old = -np.inf
    for _ in range(max_iter):
        log_resp = np.zeros((n, K))
        for k in range(K):
            diff = X - mu[k]
            log_resp[:, k] = -0.5 * np.sum(diff**2 / sigma2[k] + np.log(sigma2[k]), axis=1) + np.log(pi[k] + 1e-12)

        mx = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - mx)
        resp /= resp.sum(axis=1, keepdims=True)

        ll_new = float(np.sum(np.log(np.sum(np.exp(log_resp - mx), axis=1)) + mx.ravel()))
        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

        Nk = resp.sum(axis=0) + 1e-12
        pi = Nk / n
        for k in range(K):
            mu[k] = resp[:, k] @ X / Nk[k]
            diff = X - mu[k]
            sigma2[k] = (resp[:, k] @ (diff**2)) / Nk[k] + 1e-6

    classes = resp.argmax(axis=1)
    n_params = K - 1 + K * p + K * p
    bic = -2 * ll_new + n_params * np.log(n)

    return DescriptiveResult(
        name="lpa",
        value=float(bic),
        extra={
            "n_profiles": K,
            "log_likelihood": float(ll_new),
            "profile_sizes": [int(np.sum(classes == k)) for k in range(K)],
            "n": n,
            "p": p,
        },
    )


lpa = latent_profile


def cheatsheet() -> str:
    return "latent_profile({}) -> Latent profile analysis (GMM for continuous data)."
