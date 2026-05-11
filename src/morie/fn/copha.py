# morie.fn — function file (hadesllm/morie)
"""Cophenetic correlation."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist

from ._containers import DescriptiveResult


def cophenetic_correlation(
    data: np.ndarray,
    method: str = "average",
) -> DescriptiveResult:
    """Cophenetic correlation coefficient for hierarchical clustering quality.

    Measures how faithfully a dendrogram preserves pairwise distances.

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    method : str
        Linkage method.

    Returns
    -------
    DescriptiveResult
        ``value`` is the cophenetic correlation (0 to 1).
        ``extra`` has ``cophenetic_distances``.
    """
    X = np.asarray(data, dtype=np.float64)
    dist_vec = pdist(X)
    Z = linkage(dist_vec, method=method)
    c, coph_dists = cophenet(Z, dist_vec)

    return DescriptiveResult(
        name="CopheneticCorrelation",
        value=float(c),
        extra={"cophenetic_distances": coph_dists},
    )


copha = cophenetic_correlation


def cheatsheet() -> str:
    return "cophenetic_correlation({}) -> Cophenetic correlation for dendrograms."
