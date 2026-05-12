"""UMAP (Uniform Manifold Approximation and Projection) -- simplified."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from morie.fn._containers import UmapRes


def _fuzzy_simplicial_set(D: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Build fuzzy simplicial set (symmetrised kNN graph weights)."""
    n = D.shape[0]
    W = np.zeros((n, n))

    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:n_neighbors]
        rho = dists[nn_idx[0]]  # distance to nearest neighbour

        # Binary search for sigma
        lo, hi = 1e-10, 100.0
        sigma = 1.0
        target = np.log2(n_neighbors)
        for _ in range(64):
            vals = np.exp(-np.maximum(dists[nn_idx] - rho, 0.0) / sigma)
            s = np.sum(vals)
            if abs(s - target) < 1e-5:
                break
            if s > target:
                hi = sigma
            else:
                lo = sigma
            sigma = (lo + hi) / 2.0

        for j_pos, j in enumerate(nn_idx):
            W[i, j] = np.exp(-max(dists[j] - rho, 0.0) / sigma)

    # Symmetrise: W_sym = W + W^T - W * W^T
    W = W + W.T - W * W.T
    return W


def _spectral_init(W: np.ndarray, n_dims: int) -> np.ndarray:
    """Spectral initialisation from the graph Laplacian."""
    D_diag = np.sum(W, axis=1)
    D_diag[D_diag == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag))
    L_norm = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_norm)
    # Skip first (trivial) eigenvector
    init = eigvecs[:, 1 : n_dims + 1]
    # Scale
    init = init / (np.std(init, axis=0) + 1e-10) * 1e-4
    return init


def umap_(
    data: np.ndarray,
    n_dims: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_epochs: int = 200,
    lr: float = 1.0,
    seed: int | None = 42,
) -> UmapRes:
    """UMAP dimensionality reduction.

    Simplified implementation: builds a fuzzy simplicial set from k-NN
    distances, initialises via spectral embedding, then optimises layout
    with SGD on the cross-entropy objective.

    Parameters
    ----------
    data : ndarray (n, p)
        High-dimensional data.
    n_dims : int
        Embedding dimensionality.
    n_neighbors : int
        Number of nearest neighbours for graph construction.
    min_dist : float
        Minimum distance in embedding (controls tightness of clusters).
    n_epochs : int
        SGD epochs.
    lr : float
        Learning rate.
    seed : int, optional
        Random seed.

    Returns
    -------
    UmapRes
        ``embedding`` (n x n_dims).

    References
    ----------
    McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold
    Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    D = squareform(pdist(X, metric="euclidean"))
    W = _fuzzy_simplicial_set(D, min(n_neighbors, n - 1))

    # Smooth min_dist -> a, b parameters for the output metric
    # Approximate: phi(d) = 1 / (1 + a * d^(2b))
    # For min_dist ~ 0.1: a ~ 1.93, b ~ 0.79 (from UMAP paper fitting)
    a = 1.93 if min_dist <= 0.1 else 1.0 / min_dist
    b = 0.79

    # Spectral init (fallback to random)
    try:
        Y = _spectral_init(W, n_dims)
    except Exception:
        Y = rng.standard_normal((n, n_dims)) * 1e-4

    # Edge list (positive edges)
    rows, cols = np.where(W > 0)
    weights = W[rows, cols]

    # SGD
    for epoch in range(n_epochs):
        alpha = lr * (1.0 - epoch / n_epochs)
        for edge_idx in range(len(rows)):
            i, j = rows[edge_idx], cols[edge_idx]
            w = weights[edge_idx]

            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff**2) + 1e-10
            grad_coeff = -2.0 * a * b * dist_sq ** (b - 1.0) / (1.0 + a * dist_sq**b)
            grad = w * grad_coeff * diff

            Y[i] += alpha * grad
            Y[j] -= alpha * grad

            # Negative sample (repulsion)
            k = rng.integers(0, n)
            if k != i:
                diff_neg = Y[i] - Y[k]
                dist_sq_neg = np.sum(diff_neg**2) + 1e-10
                grad_neg = 2.0 * b / (dist_sq_neg * (1.0 + a * dist_sq_neg**b) + 1e-10)
                grad_neg = min(grad_neg, 4.0)  # clip
                Y[i] += alpha * grad_neg * diff_neg

    return UmapRes(embedding=Y)


def cheatsheet() -> str:
    return "_fuzzy_simplicial_set({}) -> UMAP (Uniform Manifold Approximation and Projection) -- simpl"
