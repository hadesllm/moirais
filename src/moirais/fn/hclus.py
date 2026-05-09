# moirais.fn — function file (hadesllm/moirais)
"""Hierarchical clustering."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ._containers import HclstRes


def hierarchical_cluster(
    data: np.ndarray,
    n_clusters: int = 3,
    method: str = "ward",
    metric: str = "euclidean",
) -> HclstRes:
    """Agglomerative hierarchical clustering.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    n_clusters : int
        Number of clusters to cut.
    method : str
        Linkage method: ``'single'``, ``'complete'``, ``'average'``, ``'ward'``.
    metric : str
        Distance metric (ignored for Ward, which uses ``'euclidean'``).

    Returns
    -------
    HclstRes
    """
    X = np.asarray(data, dtype=np.float64)

    if method == "ward":
        metric = "euclidean"

    dist_vec = pdist(X, metric=metric)
    Z = linkage(dist_vec, method=method, metric=metric)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

    return HclstRes(
        labels=labels,
        linkage_matrix=Z,
        distances=dist_vec,
    )


hclus = hierarchical_cluster


def cheatsheet() -> str:
    return "hierarchical_cluster({}) -> Agglomerative hierarchical clustering."
