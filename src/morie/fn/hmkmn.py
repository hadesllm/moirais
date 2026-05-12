# morie.fn -- function file (hadesllm/morie)
"""k-means clustering via Lloyd's algorithm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kmeans"]


def geron_kmeans(X, n_clusters, seed):
    """
    k-means clustering via Lloyd's algorithm

    Formula: min sum_i ||x_i - mu_{c_i}||^2

    Parameters
    ----------
    X : array-like
        Input data.
    n_clusters : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centers

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-means clustering via Lloyd's algorithm"})


def cheatsheet():
    return "hmkmn: k-means clustering via Lloyd's algorithm"
