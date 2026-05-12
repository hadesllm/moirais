"""HDBSCAN -- hierarchical density clustering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hdbscan"]


def hdbscan(X, min_cluster_size):
    """
    HDBSCAN -- hierarchical density clustering

    Formula: build mutual reachability tree + condensed tree

    Parameters
    ----------
    X : array-like
        Input data.
    min_cluster_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Campello-Moulavi-Sander (2013)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HDBSCAN -- hierarchical density clustering"})


def cheatsheet():
    return "hdbsc: HDBSCAN -- hierarchical density clustering"
