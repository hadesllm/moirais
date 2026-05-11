# morie.fn — function file (hadesllm/morie)
"""No man ever steps in the same river twice. — Heraclitus"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ._containers import DescriptiveResult


def dbscan_dr(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> DescriptiveResult:
    """
    DBSCAN clustering returning DescriptiveResult with diagnostics.

    :param X: Data matrix (n_samples, n_features).
    :param eps: Maximum distance between neighbours. Default 0.5.
    :param min_samples: Minimum points to form a dense region. Default 5.
    :return: DescriptiveResult with number of clusters as value.
    :raises ValueError: If X is not 2-D, or eps/min_samples invalid.

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based
    algorithm for discovering clusters in large spatial databases with noise.
    KDD-96, 226--231.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array.")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}.")

    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    dist = cdist(X, X)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = list(np.where(dist[i] <= eps)[0])
        if len(neighbours) < min_samples:
            continue
        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbours = list(np.where(dist[q] <= eps)[0])
                if len(q_neighbours) >= min_samples:
                    seed_set.extend(q_neighbours)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1

    n_noise = int(np.sum(labels == -1))

    return DescriptiveResult(
        name="DBSCAN",
        value=cluster_id,
        extra={
            "labels": labels,
            "n_clusters": cluster_id,
            "n_noise": n_noise,
            "eps": eps,
            "min_samples": min_samples,
            "n_samples": n,
        },
    )


dbscr = dbscan_dr


def cheatsheet() -> str:
    return "dbscan_dr({}) -> DBSCAN clustering (DescriptiveResult). 'I've got a good feel"
