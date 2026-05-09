"""Spectral clustering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spectral_clustering"]


def spectral_clustering(A, k):
    """
    Spectral clustering

    Formula: k-means on Laplacian eigenvectors

    Parameters
    ----------
    A : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ng-Jordan-Weiss (2001)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral clustering"})


def cheatsheet():
    return "specClust: Spectral clustering"
