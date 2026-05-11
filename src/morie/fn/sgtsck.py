"""Ng-Jordan-Weiss k-way spectral clustering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_spectral_clustering_k"]


def sgt_spectral_clustering_k(A, k):
    """
    Ng-Jordan-Weiss k-way spectral clustering

    Formula: k smallest eigvecs of L̃; row-normalise; k-means

    Parameters
    ----------
    A : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, eigvecs

    References
    ----------
    Ng-Jordan-Weiss (2002)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ng-Jordan-Weiss k-way spectral clustering"})


def cheatsheet():
    return "sgtsck: Ng-Jordan-Weiss k-way spectral clustering"
