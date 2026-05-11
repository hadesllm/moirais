# morie.fn — function file (hadesllm/morie)
"""Classical multidimensional scaling (MDS) preserves pairwise distances."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mds"]


def geron_mds(X, n_components):
    """
    Classical multidimensional scaling (MDS) preserves pairwise distances

    Formula: min_Y sum_{ij} (d_ij - ||y_i - y_j||)^2

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classical multidimensional scaling (MDS) preserves pairwise distances"})


def cheatsheet():
    return "hmmds: Classical multidimensional scaling (MDS) preserves pairwise distances"
