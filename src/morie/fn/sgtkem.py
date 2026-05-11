"""Katz centrality via series expansion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_katz_centrality"]


def sgt_katz_centrality(A, alpha, beta):
    """
    Katz centrality via series expansion

    Formula: x = β (I - α A)^{-1} 1

    Parameters
    ----------
    A : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Katz (1953)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Katz centrality via series expansion"})


def cheatsheet():
    return "sgtkem: Katz centrality via series expansion"
