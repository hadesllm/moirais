# morie.fn -- function file (hadesllm/morie)
"""Isomap embedding. 'Truly wonderful the mind of a child is.'"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def isomap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 10,
) -> DescriptiveResult:
    """Isomap nonlinear dimensionality reduction.

    Constructs a k-nearest-neighbor graph, computes shortest-path
    (geodesic) distances via Floyd-Warshall, then applies classical
    MDS to embed into the target dimensionality.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    n_components : int
        Target embedding dimensionality.
    n_neighbors : int
        Number of nearest neighbors for the graph.

    Returns
    -------
    DescriptiveResult
        name='Isomap', value=residual variance,
        extra has 'embedding' (ndarray), 'geodesic_distances' (ndarray),
        'n_neighbors', 'n_components'.

    References
    ----------
    Tenenbaum, J.B., de Silva, V. & Langford, J.C. (2000). A global
    geometric framework for nonlinear dimensionality reduction.
    *Science*, 290(5500), 2319-2323. doi:10.1126/science.290.5500.2319
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = min(n_neighbors, n - 1)

    D = cdist(X, X, "euclidean")

    G = np.full((n, n), np.inf)
    np.fill_diagonal(G, 0.0)
    for i in range(n):
        neighbors = np.argsort(D[i])[: k + 1]
        for j in neighbors:
            G[i, j] = D[i, j]
            G[j, i] = D[j, i]

    for mid in range(n):
        new_dist = G[:, mid, None] + G[None, mid, :]
        G = np.minimum(G, new_dist)

    G_sq = G**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ G_sq @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals_sel = np.maximum(eigvals[idx], 0.0)
    embedding = eigvecs[:, idx] * np.sqrt(eigvals_sel)[None, :]

    eucl_dist = cdist(X, X, "euclidean")
    corr = np.corrcoef(G.ravel(), eucl_dist.ravel())[0, 1]
    residual_var = max(0.0, 1.0 - corr**2)

    return DescriptiveResult(
        name="Isomap",
        value=float(residual_var),
        extra={
            "embedding": embedding,
            "geodesic_distances": G,
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "n_samples": n,
        },
    )


def cheatsheet() -> str:
    return "isomap({}) -> Isomap embedding. 'Truly wonderful the mind of a child is.' "
