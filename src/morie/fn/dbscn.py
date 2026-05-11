# morie.fn — function file (hadesllm/morie)
"""DBSCAN density-based clustering."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from morie.fn._containers import DbscnRes


def dbscn(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> DbscnRes:
    """DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Expands clusters from core points whose eps-neighbourhoods contain
    at least *min_samples* points.  Points not reachable from any core
    point are labelled as noise (-1).

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    eps : float
        Neighbourhood radius.
    min_samples : int
        Minimum points to form a dense region.

    Returns
    -------
    DbscnRes
        ``labels`` (-1 for noise), ``n_clusters``, ``n_noise``.

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based
    algorithm for discovering clusters in large spatial databases with noise.
    *Proc. KDD*, 226-231.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    D = squareform(pdist(X, metric="euclidean"))

    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbours = np.where(D[i] <= eps)[0]
        if len(neighbours) < min_samples:
            # Noise (may be reassigned later)
            continue

        # Start new cluster
        labels[i] = cluster_id
        seed_set = list(neighbours)
        seed_set.remove(i)

        while seed_set:
            j = seed_set.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbours = np.where(D[j] <= eps)[0]
                if len(j_neighbours) >= min_samples:
                    seed_set.extend(k for k in j_neighbours if not visited[k])
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    n_noise = int(np.sum(labels == -1))
    return DbscnRes(
        labels=labels,
        n_clusters=cluster_id,
        n_noise=n_noise,
    )


def cheatsheet() -> str:
    return "dbscn({}) -> DBSCAN density-based clustering."
