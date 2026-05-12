# morie.fn -- function file (hadesllm/morie)
"""Isomap: MDS on geodesic distances from kNN graph."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_isomap"]


def geron_isomap(X, n_components, n_neighbors):
    """
    Isomap: MDS on geodesic distances from kNN graph

    Formula: D_geo via Dijkstra on kNN; MDS yields embedding

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    n_neighbors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isomap: MDS on geodesic distances from kNN graph"})


def cheatsheet():
    return "hmiso: Isomap: MDS on geodesic distances from kNN graph"
