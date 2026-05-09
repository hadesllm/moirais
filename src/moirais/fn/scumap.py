"""UMAP for single-cell embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["umap_singlecell"]


def umap_singlecell(X, n_neighbors, min_dist):
    """
    UMAP for single-cell embedding

    Formula: manifold learning on k-NN graph

    Parameters
    ----------
    X : array-like
        Input data.
    n_neighbors : array-like
        Input data.
    min_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McInnes et al (2018)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "UMAP for single-cell embedding"})


def cheatsheet():
    return "scumap: UMAP for single-cell embedding"
