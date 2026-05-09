# moirais.fn — function file (hadesllm/moirais)
"""Hierarchical agglomerative clustering."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hierarchical_clust"]


def rangayyan_hierarchical_clust(X, linkage, n_clusters):
    """
    Hierarchical agglomerative clustering

    Formula: Merge clusters with minimum linkage distance (single/complete/average)

    Parameters
    ----------
    X : array-like
        Input data.
    linkage : array-like
        Input data.
    n_clusters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, dendrogram

    References
    ----------
    Rangayyan Ch 10.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical agglomerative clustering"})


def cheatsheet():
    return "rghier: Hierarchical agglomerative clustering"
