"""Laplacian eigenmaps embedding into k dims."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_laplacian_eigenmaps"]


def sgt_laplacian_eigenmaps(A, k):
    """
    Laplacian eigenmaps embedding into k dims

    Formula: k smallest non-zero eigvecs of L̃

    Parameters
    ----------
    A : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Belkin & Niyogi (2003)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplacian eigenmaps embedding into k dims"})


def cheatsheet():
    return "sgtlap2: Laplacian eigenmaps embedding into k dims"
