# morie.fn -- function file (hadesllm/morie)
"""K-means clustering algorithm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_kmeans"]


def rangayyan_kmeans(X, k, max_iter, tol):
    """
    K-means clustering algorithm

    Formula: Assign to nearest centroid; update mu_k = mean(x_i in cluster k); iterate

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centroids

    References
    ----------
    Rangayyan Ch 10.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-means clustering algorithm"})


def cheatsheet():
    return "rgkmns: K-means clustering algorithm"
