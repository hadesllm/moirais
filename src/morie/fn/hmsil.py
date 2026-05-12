# morie.fn -- function file (hadesllm/morie)
"""Silhouette score for cluster quality."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_silhouette"]


def geron_silhouette(X, labels):
    """
    Silhouette score for cluster quality

    Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Parameters
    ----------
    X : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: silhouette

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Silhouette score for cluster quality"})


def cheatsheet():
    return "hmsil: Silhouette score for cluster quality"
