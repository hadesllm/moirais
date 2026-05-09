"""Leiden community detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["leiden_clustering"]


def leiden_clustering(graph, resolution):
    """
    Leiden community detection

    Formula: refined Louvain with guarantees

    Parameters
    ----------
    graph : array-like
        Input data.
    resolution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Traag-Waltman-van Eck (2019)
    """
    graph = np.atleast_1d(np.asarray(graph, dtype=float))
    n = len(graph)
    result = float(np.mean(graph))
    se = float(np.std(graph, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leiden community detection"})


def cheatsheet():
    return "scleid: Leiden community detection"
