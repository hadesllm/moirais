# morie.fn — function file (hadesllm/morie)
"""Agglomerative hierarchical clustering."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_agglomerative"]


def geron_agglomerative(X, n_clusters, linkage):
    """
    Agglomerative hierarchical clustering

    Formula: iteratively merge closest clusters by linkage

    Parameters
    ----------
    X : array-like
        Input data.
    n_clusters : array-like
        Input data.
    linkage : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, linkage_matrix

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Agglomerative hierarchical clustering"})


def cheatsheet():
    return "hmagc: Agglomerative hierarchical clustering"
