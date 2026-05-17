"""Spectral clustering using an RBF affinity kernel."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rbf_spectral_cluster(
    X: np.ndarray,
    *,
    k: int = 3,
    sigma: float | None = None,
    n_iter: int = 100,
    seed: int | None = None,
) -> DescriptiveResult:
    """Spectral clustering using an RBF affinity kernel.

    Builds a similarity graph with Gaussian kernel, computes the normalised
    Laplacian eigenvectors, then clusters with k-means in the embedding space.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n x p).
    k : int
        Number of clusters.
    sigma : float or None
        RBF bandwidth. If None, uses median pairwise distance heuristic.
    n_iter : int
        Max k-means iterations.
    seed : int or None
        RNG seed.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``labels`` (n,) and ``eigenvalues`` (k,).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n = X.shape[0]
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, {n}]")

    dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    if sigma is None:
        sigma = float(np.sqrt(np.median(dists[dists > 0])))
    if sigma <= 0:
        sigma = 1.0

    W = np.exp(-dists / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    deg = W.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L)
    U = eigvecs[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = U / norms

    rng = np.random.default_rng(seed)
    centers = U[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.array([np.linalg.norm(U - c, axis=1) for c in centers])
        new_labels = d.argmin(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = U[mask].mean(axis=0)

    return DescriptiveResult(
        name="rbf_spectral_cluster",
        value={"labels": labels, "eigenvalues": eigvals[:k]},
        extra={"k": k, "sigma": sigma, "n": n},
    )


speclu = rbf_spectral_cluster


def cheatsheet() -> str:
    return 'rbf_spectral_cluster({}) -> Spectral clustering variant.'
