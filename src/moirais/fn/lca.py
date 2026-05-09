# moirais.fn — function file (hadesllm/moirais)
"""Latent class analysis (EM for binary indicators)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def latent_class(
    X_binary: np.ndarray,
    n_classes: int = 2,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int | None = None,
) -> DescriptiveResult:
    """Latent class analysis via EM for binary indicators.

    Parameters
    ----------
    X_binary : (n, p) binary array
    n_classes : int
    max_iter, tol : convergence controls
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X_binary, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    K = n_classes
    rng = np.random.default_rng(seed)

    pi = np.ones(K) / K
    theta = rng.uniform(0.2, 0.8, size=(K, p))

    ll_old = -np.inf
    for _ in range(max_iter):
        log_lik = np.zeros((n, K))
        for k in range(K):
            log_lik[:, k] = np.sum(
                X * np.log(theta[k] + 1e-12) + (1 - X) * np.log(1 - theta[k] + 1e-12),
                axis=1,
            ) + np.log(pi[k] + 1e-12)

        log_lik_max = log_lik.max(axis=1, keepdims=True)
        resp = np.exp(log_lik - log_lik_max)
        resp /= resp.sum(axis=1, keepdims=True)

        ll_new = float(np.sum(np.log(np.sum(np.exp(log_lik - log_lik_max), axis=1)) + log_lik_max.ravel()))
        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

        Nk = resp.sum(axis=0)
        pi = Nk / n
        for k in range(K):
            theta[k] = (resp[:, k] @ X) / (Nk[k] + 1e-12)
            theta[k] = np.clip(theta[k], 0.01, 0.99)

    classes = resp.argmax(axis=1)
    bic = -2 * ll_new + (K - 1 + K * p) * np.log(n)

    return DescriptiveResult(
        name="lca",
        value=float(bic),
        extra={
            "n_classes": K,
            "log_likelihood": float(ll_new),
            "class_priors": pi.tolist(),
            "class_sizes": [int(np.sum(classes == k)) for k in range(K)],
            "n": n,
            "p": p,
        },
    )


lca = latent_class


def cheatsheet() -> str:
    return "latent_class({}) -> Latent class analysis (EM for binary indicators)."
