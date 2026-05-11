# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Agglomerative hierarchical clustering. 'Let the past die.' -- Kylo Ren"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from ._containers import DescriptiveResult


def agglomerative(
    X: np.ndarray,
    n_clusters: int = 3,
    method: str = "ward",
) -> DescriptiveResult:
    """
    Agglomerative (bottom-up) hierarchical clustering.

    :param X: Data matrix (n_samples, n_features).
    :param n_clusters: Number of clusters to extract. Default 3.
    :param method: Linkage method: 'ward', 'single', 'complete', 'average'.
    :return: DescriptiveResult with cluster labels and linkage matrix.
    :raises ValueError: If X not 2-D or n_clusters invalid.

    References
    ----------
    Ward, J. H. (1963). Hierarchical grouping to optimize an objective
    function. Journal of the American Statistical Association, 58(301),
    236--244. doi:10.1080/01621459.1963.10500845
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2-D array with >= 2 samples.")
    if not 1 <= n_clusters <= X.shape[0]:
        raise ValueError(f"n_clusters must be in [1, {X.shape[0]}], got {n_clusters}.")

    Z = linkage(X, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

    unique, counts = np.unique(labels, return_counts=True)

    return DescriptiveResult(
        name="Agglomerative Clustering",
        value=n_clusters,
        extra={
            "labels": labels,
            "linkage_matrix": Z,
            "n_clusters": n_clusters,
            "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "method": method,
            "n_samples": X.shape[0],
        },
    )


aggcl = agglomerative


def cheatsheet() -> str:
    return "agglomerative({}) -> Agglomerative hierarchical clustering. 'Let the past die.' -"
