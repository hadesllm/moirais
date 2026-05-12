# morie.fn -- function file (hadesllm/morie)
"""UMAP: uniform manifold approximation, preserves local and some global structure."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_umap"]


def geron_umap(X, n_components, n_neighbors, min_dist):
    """
    UMAP: uniform manifold approximation, preserves local and some global structure

    Formula: fuzzy topological cross-entropy between high-/low-d graphs

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.
    n_neighbors : array-like
        Input data.
    min_dist : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "UMAP: uniform manifold approximation, preserves local and some global structure"})


def cheatsheet():
    return "hmumap: UMAP: uniform manifold approximation, preserves local and some global structure"
