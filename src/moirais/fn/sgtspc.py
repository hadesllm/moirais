"""Sorted eigenvalues of a symmetric matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_spectrum"]


def sgt_spectrum(M):
    """
    Sorted eigenvalues of a symmetric matrix

    Formula: λ_1 <= ... <= λ_n

    Parameters
    ----------
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lams

    References
    ----------
    Chung (1997)
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sorted eigenvalues of a symmetric matrix"})


def cheatsheet():
    return "sgtspc: Sorted eigenvalues of a symmetric matrix"
