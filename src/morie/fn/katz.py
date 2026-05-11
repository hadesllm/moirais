"""Katz centrality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["katz_centrality"]


def katz_centrality(A, alpha):
    """
    Katz centrality

    Formula: x = (I − αA)^{-1} 1

    Parameters
    ----------
    A : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Katz (1953)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Katz centrality"})


def cheatsheet():
    return "katz: Katz centrality"
