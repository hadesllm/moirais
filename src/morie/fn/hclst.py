# morie.fn -- function file (hadesllm/morie)
"""Hierarchical agglomerative clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from morie.fn._containers import HclstRes


def hclst(
    data: pd.DataFrame | np.ndarray,
    k: int = 3,
    method: str = "ward",
) -> HclstRes:
    """Hierarchical agglomerative clustering.

    Computes a linkage matrix using the specified method and cuts the
    dendrogram at *k* clusters.

    Parameters
    ----------
    data : DataFrame or ndarray
        Observations as rows.
    k : int
        Number of clusters to extract.
    method : str
        Linkage method: ``"ward"`` (default), ``"complete"``, ``"average"``,
        ``"single"``.

    Returns
    -------
    HclstRes
        ``labels``, ``linkage_matrix``, ``distances``.

    References
    ----------
    Ward, J. H. (1963). Hierarchical grouping to optimize an objective
    function. *JASA*, 58(301), 236-244.  DOI: 10.1080/01621459.1963.10500845
    """
    X = np.asarray(data, dtype=np.float64)
    distances = pdist(X, metric="euclidean")
    Z = linkage(distances, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    # fcluster labels start at 1; shift to 0-based
    labels = labels - 1

    return HclstRes(
        labels=labels,
        linkage_matrix=Z,
        distances=distances,
    )


def cheatsheet() -> str:
    return "hclst({}) -> Hierarchical agglomerative clustering."
