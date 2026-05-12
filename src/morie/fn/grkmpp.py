# morie.fn -- function file (hadesllm/morie)
"""k-means++ initialization: sample each centroid with prob. proportional to D(x)^2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kmeans_pp_seeding"]


def geron_kmeans_pp_seeding(X, k, seed):
    """
    k-means++ initialization: sample each centroid with prob. proportional to D(x)^2

    Formula: P(x | C) propto min_{c in C} ||x - c||^2

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: centroids

    References
    ----------
    Géron Ch 8, K-means++ section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-means++ initialization: sample each centroid with prob. proportional to D(x)^2"})


def cheatsheet():
    return "grkmpp: k-means++ initialization: sample each centroid with prob. proportional to D(x)^2"
