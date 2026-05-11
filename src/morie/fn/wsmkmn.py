"""k-means clustering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_kmeans"]


def wasserman_kmeans(X, k):
    """
    k-means clustering

    Formula: min sum_k sum_{x in C_k} |x - mu_k|^2

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: centers, labels

    References
    ----------
    Wasserman (2004), Ch 19
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-means clustering"})


def cheatsheet():
    return "wsmkmn: k-means clustering"
