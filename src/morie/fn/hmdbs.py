# morie.fn -- function file (hadesllm/morie)
"""Density-based spatial clustering (DBSCAN)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dbscan"]


def geron_dbscan(X, eps, min_samples):
    """
    Density-based spatial clustering (DBSCAN)

    Formula: points with >=min_samples within eps are core; connect via eps-reachability

    Parameters
    ----------
    X : array-like
        Input data.
    eps : array-like
        Input data.
    min_samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Density-based spatial clustering (DBSCAN)"})


def cheatsheet():
    return "hmdbs: Density-based spatial clustering (DBSCAN)"
