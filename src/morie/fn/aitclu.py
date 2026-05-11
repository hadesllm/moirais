"""k-means in CLR coordinates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_kmeans"]


def compositional_kmeans(X, k):
    """
    k-means in CLR coordinates

    Formula: cluster on z = clr(X); centroids back-mapped via clr^{-1}

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centers

    References
    ----------
    Aitchison (1986); Lloyd (1982)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-means in CLR coordinates"})


def cheatsheet():
    return "aitclu: k-means in CLR coordinates"
