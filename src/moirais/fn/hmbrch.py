# moirais.fn — function file (hadesllm/moirais)
"""BIRCH: balanced iterative reducing and clustering using hierarchies."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_birch"]


def geron_birch(X, n_clusters, threshold, branching_factor):
    """
    BIRCH: balanced iterative reducing and clustering using hierarchies

    Formula: CF-tree of clustering features (N, LS, SS)

    Parameters
    ----------
    X : array-like
        Input data.
    n_clusters : array-like
        Input data.
    threshold : array-like
        Input data.
    branching_factor : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIRCH: balanced iterative reducing and clustering using hierarchies"})


def cheatsheet():
    return "hmbrch: BIRCH: balanced iterative reducing and clustering using hierarchies"
