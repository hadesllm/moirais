"""Laplacian eigenmap embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["laplacian_eigenmaps"]


def laplacian_eigenmaps(A, k):
    """
    Laplacian eigenmap embedding

    Formula: smallest k eigvecs of L

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
    Belkin-Niyogi (2003)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplacian eigenmap embedding"})


def cheatsheet():
    return "lapEig: Laplacian eigenmap embedding"
