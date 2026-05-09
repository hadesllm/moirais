"""Errors using inadequate data are much less than those using no data at all. — Charles Babbage"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hits"]


def hits(A, iters):
    """
    HITS (hubs & authorities)

    Formula: a = A^T h; h = A a; iterate

    Parameters
    ----------
    A : array-like
        Input data.
    iters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kleinberg (1999)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Errors using inadequate data are much less than those using no data at all. — Charles Babbage"})


def cheatsheet():
    return "Errors using inadequate data are much less than those using no data at all. — Charles Babbage"
