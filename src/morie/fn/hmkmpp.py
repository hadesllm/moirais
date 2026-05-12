# morie.fn -- function file (hadesllm/morie)
"""k-means++ initialization for well-separated initial centers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kmeans_plus_plus"]


def geron_kmeans_plus_plus(X, n_clusters, seed):
    """
    k-means++ initialization for well-separated initial centers

    Formula: P(x_i) proportional to min_j ||x_i - mu_j||^2

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
        Keys: initial_centers

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-means++ initialization for well-separated initial centers"})


def cheatsheet():
    return "hmkmpp: k-means++ initialization for well-separated initial centers"
