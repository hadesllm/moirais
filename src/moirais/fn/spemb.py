"""Spectral embedding via Laplacian eigenmaps. 'Hope is not lost today.' -- Poe Dameron"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def spectral_embed(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 10,
    sigma: float | None = None,
) -> DescriptiveResult:
    """Spectral embedding via the normalized graph Laplacian.

    Constructs a k-nearest-neighbor affinity graph, computes the
    normalized Laplacian, and embeds using its smallest non-trivial
    eigenvectors (Laplacian eigenmaps).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    n_components : int
        Number of embedding dimensions.
    n_neighbors : int
        Number of nearest neighbors for graph construction.
    sigma : float or None
        Gaussian kernel bandwidth. If None, uses median neighbor distance.

    Returns
    -------
    DescriptiveResult
        name='Spectral Embedding', value=n_samples,
        extra has 'embedding' (ndarray), 'eigenvalues',
        'sigma', 'n_neighbors', 'n_components'.

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for
    Dimensionality Reduction and Data Representation. *Neural
    Computation*, 15(6), 1373-1396. doi:10.1162/089976603321780317
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = min(n_neighbors, n - 1)

    D = cdist(X, X, "euclidean")
    np.fill_diagonal(D, np.inf)
    knn_idx = np.argsort(D, axis=1)[:, :k]

    if sigma is None:
        knn_dists = np.take_along_axis(D, knn_idx, axis=1)
        sigma = float(np.median(knn_dists))
        if sigma < 1e-10:
            sigma = 1.0

    np.fill_diagonal(D, 0.0)
    W = np.zeros((n, n))
    for i in range(n):
        for j in knn_idx[i]:
            val = np.exp(-(D[i, j] ** 2) / (2.0 * sigma**2))
            W[i, j] = val
            W[j, i] = val

    degree = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_norm)
    idx = np.argsort(eigvals)[1 : n_components + 1]
    embedding = eigvecs[:, idx]

    return DescriptiveResult(
        name="Spectral Embedding",
        value=n,
        extra={
            "embedding": embedding,
            "eigenvalues": eigvals[idx],
            "sigma": sigma,
            "n_neighbors": k,
            "n_components": n_components,
        },
    )


spemb = spectral_embed


def cheatsheet() -> str:
    return "spectral_embed({}) -> Spectral embedding via Laplacian eigenmaps. 'Hope is not los"
