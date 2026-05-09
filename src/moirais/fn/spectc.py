"""Spectral clustering. 'Be with me.' -- Rey"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import eigsh

from ._containers import DescriptiveResult


def spectral_cluster(
    affinity: np.ndarray,
    n_clusters: int = 3,
    seed: int = 42,
) -> DescriptiveResult:
    """
    Spectral clustering via the normalised graph Laplacian.

    Steps: (1) compute normalised Laplacian, (2) embed in the space of
    the *k* smallest eigenvectors, (3) run k-means.

    :param affinity: Symmetric affinity/similarity matrix (n x n).
    :param n_clusters: Number of clusters. Default 3.
    :param seed: RNG seed for k-means. Default 42.
    :return: DescriptiveResult with cluster labels.
    :raises ValueError: If affinity not square-symmetric or n_clusters invalid.

    References
    ----------
    Ng, A. Y., Jordan, M. I., & Weiss, Y. (2001). On spectral clustering:
    analysis and an algorithm. NIPS 14, 849--856.
    """
    A = np.asarray(affinity, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("affinity must be a square 2-D array.")
    n = A.shape[0]
    if not 2 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [2, {n}], got {n_clusters}.")

    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-12)))
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    L_sparse = None
    try:
        from scipy.sparse import csc_matrix

        L_sparse = csc_matrix(L_sym)
        vals, vecs = eigsh(L_sparse, k=n_clusters, which="SM")
    except Exception:
        vals_all, vecs_all = np.linalg.eigh(L_sym)
        vals = vals_all[:n_clusters]
        vecs = vecs_all[:, :n_clusters]

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embedding = vecs / norms

    centroids, labels = kmeans2(embedding, n_clusters, minit="points", seed=seed)

    unique, counts = np.unique(labels, return_counts=True)

    return DescriptiveResult(
        name="Spectral Clustering",
        value=n_clusters,
        extra={
            "labels": labels,
            "eigenvalues": vals.tolist(),
            "embedding": embedding,
            "n_clusters": n_clusters,
            "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "n_samples": n,
        },
    )


spectc = spectral_cluster


def cheatsheet() -> str:
    return "spectral_cluster({}) -> Spectral clustering. 'Be with me.' -- Rey"
