# morie.fn -- function file (hadesllm/morie)
"""Probabilistic PCA via EM."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def probabilistic_pca(
    X: np.ndarray,
    n_components: int = 2,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int | None = None,
) -> DescriptiveResult:
    """Probabilistic PCA (Tipping & Bishop, 1999) via EM.

    Parameters
    ----------
    X : (n, p) array
    n_components : int
    max_iter, tol : convergence controls
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    q = min(n_components, p)

    mu = X.mean(axis=0)
    X_c = X - mu
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((p, q))
    sigma2 = 1.0

    for _ in range(max_iter):
        M = W.T @ W + sigma2 * np.eye(q)
        M_inv = np.linalg.inv(M)
        Ez = X_c @ W @ M_inv.T
        Ezz = sigma2 * M_inv + Ez[:, :, None] * Ez[:, None, :]

        W_new = (X_c.T @ Ez) @ np.linalg.inv(Ezz.sum(axis=0))
        sigma2_new = (np.sum(X_c**2) - 2 * np.sum(Ez * (X_c @ W_new)) + np.sum(Ezz * (W_new.T @ W_new)[None])) / (n * p)
        sigma2_new = max(sigma2_new, 1e-10)

        if np.max(np.abs(W_new - W)) < tol:
            W = W_new
            sigma2 = sigma2_new
            break
        W = W_new
        sigma2 = sigma2_new

    var_explained = np.sum(W**2, axis=0)

    return DescriptiveResult(
        name="ppca",
        value=float(sigma2),
        extra={
            "W_shape": list(W.shape),
            "sigma2": float(sigma2),
            "var_explained": var_explained.tolist(),
            "n": n,
            "p": p,
            "q": q,
        },
    )


ppca = probabilistic_pca


def cheatsheet() -> str:
    return "probabilistic_pca({}) -> Probabilistic PCA via EM."
