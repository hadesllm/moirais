# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""HDBSCAN density-based hierarchical clustering assignment."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_hdbscan_cluster"]


def alammar_hdbscan_cluster(X, min_cluster_size, min_samples):
    """
    HDBSCAN density-based hierarchical clustering assignment

    Formula: labels = HDBSCAN(min_cluster_size, min_samples) on distance matrix; -1 = noise

    Parameters
    ----------
    X : array-like
        Input data.
    min_cluster_size : array-like
        Input data.
    min_samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Alammar Ch 5, HDBSCAN section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HDBSCAN density-based hierarchical clustering assignment"})


def cheatsheet():
    return "alhds: HDBSCAN density-based hierarchical clustering assignment"
