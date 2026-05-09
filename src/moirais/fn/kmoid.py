# moirais.fn — function file (hadesllm/moirais)
"""K-medoids (PAM algorithm)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._containers import DescriptiveResult


def kmedoids(
    data: np.ndarray,
    k: int = 3,
    max_iter: int = 300,
    seed: int = 42,
) -> DescriptiveResult:
    """K-medoids clustering via Partitioning Around Medoids (PAM).

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    k : int
        Number of clusters.
    max_iter : int
        Maximum iterations.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is cluster labels.
        ``extra`` has ``medoid_indices``, ``medoids``, ``cost``.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    D = squareform(pdist(X))
    rng = np.random.default_rng(seed)

    medoid_idx = rng.choice(n, size=k, replace=False)

    for _ in range(max_iter):
        dists_to_medoids = D[:, medoid_idx]
        labels = np.argmin(dists_to_medoids, axis=1)
        cost = np.sum(dists_to_medoids[np.arange(n), labels])

        new_medoids = medoid_idx.copy()
        for j in range(k):
            cluster_mask = labels == j
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_dists = D[np.ix_(cluster_indices, cluster_indices)]
            total_dists = cluster_dists.sum(axis=1)
            best = cluster_indices[np.argmin(total_dists)]
            new_medoids[j] = best

        if np.array_equal(np.sort(new_medoids), np.sort(medoid_idx)):
            break
        medoid_idx = new_medoids

    dists_to_medoids = D[:, medoid_idx]
    labels = np.argmin(dists_to_medoids, axis=1)
    cost = float(np.sum(dists_to_medoids[np.arange(n), labels]))

    return DescriptiveResult(
        name="KMedoids",
        value=labels,
        extra={
            "medoid_indices": medoid_idx,
            "medoids": X[medoid_idx],
            "cost": cost,
        },
    )


kmoid = kmedoids


def cheatsheet() -> str:
    return "kmedoids({}) -> K-medoids (PAM) clustering."
