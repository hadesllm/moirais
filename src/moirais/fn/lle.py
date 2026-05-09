# moirais.fn — function file (hadesllm/moirais)
"""Locally Linear Embedding. 'Luminous beings are we, not this crude matter.'"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def lle(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 12,
) -> DescriptiveResult:
    """Locally Linear Embedding for nonlinear dimensionality reduction.

    Reconstructs each point as a weighted combination of its neighbors,
    then finds low-dimensional coordinates preserving those weights.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    n_components : int
        Target dimensionality.
    n_neighbors : int
        Number of nearest neighbors per point.

    Returns
    -------
    DescriptiveResult
        name='LLE', value=reconstruction error,
        extra has 'embedding' (ndarray), 'weights' (ndarray),
        'n_neighbors', 'n_components'.

    References
    ----------
    Roweis, S.T. & Saul, L.K. (2000). Nonlinear dimensionality
    reduction by locally linear embedding. *Science*, 290(5500),
    2323-2326. doi:10.1126/science.290.5500.2323
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    k = min(n_neighbors, n - 1)

    D = cdist(X, X, "euclidean")
    np.fill_diagonal(D, np.inf)
    neighbors = np.argsort(D, axis=1)[:, :k]

    W = np.zeros((n, n))
    for i in range(n):
        Z = X[neighbors[i]] - X[i]
        C = Z @ Z.T
        C += np.eye(k) * 1e-3 * np.trace(C)
        try:
            w = np.linalg.solve(C, np.ones(k))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(C, np.ones(k), rcond=None)[0]
        w /= w.sum()
        W[i, neighbors[i]] = w

    M = (np.eye(n) - W).T @ (np.eye(n) - W)

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(np.abs(eigvals))
    embedding = eigvecs[:, idx[1 : n_components + 1]]

    recon_err = float(np.trace(embedding.T @ M @ embedding))

    return DescriptiveResult(
        name="LLE",
        value=recon_err,
        extra={
            "embedding": embedding,
            "weights": W,
            "n_neighbors": k,
            "n_components": n_components,
            "n_samples": n,
        },
    )


def cheatsheet() -> str:
    return "lle({}) -> Locally Linear Embedding. 'Luminous beings are we, not this "
