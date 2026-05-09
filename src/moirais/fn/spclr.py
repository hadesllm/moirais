"""Spectral clustering."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._containers import DescriptiveResult
from .kmcls import kmeans


def spectral_clustering(
    data: np.ndarray,
    n_clusters: int = 3,
    gamma: float | None = None,
    seed: int = 42,
) -> DescriptiveResult:
    """Spectral clustering via normalised graph Laplacian.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    n_clusters : int
        Number of clusters.
    gamma : float, optional
        RBF kernel bandwidth. Defaults to 1/p.
    seed : int
        Random seed for k-means step.

    Returns
    -------
    DescriptiveResult
        ``value`` is cluster labels.
        ``extra`` has ``eigenvalues``, ``embedding``.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    if gamma is None:
        gamma = 1.0 / p

    D_sq = squareform(pdist(X, "sqeuclidean"))
    W = np.exp(-gamma * D_sq)
    np.fill_diagonal(W, 0)

    D_vec = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D_vec, 1e-12)))
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_norm)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    embedding = eigvecs[:, :n_clusters]
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    embedding /= row_norms

    km = kmeans(embedding, k=n_clusters, seed=seed)

    return DescriptiveResult(
        name="SpectralClustering",
        value=km.labels,
        extra={
            "eigenvalues": eigvals[:n_clusters + 1],
            "embedding": embedding,
        },
    )


spclr = spectral_clustering


def cheatsheet() -> str:
    return "spectral_clustering({}) -> Spectral clustering via graph Laplacian."
