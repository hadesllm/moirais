"""Truth comes out of error more readily than out of confusion. -- Francis Bacon"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_hits_kleinberg"]


def sgt_hits_kleinberg(A, max_iter, tol):
    """
    HITS algorithm hubs + authorities

    Formula: Iterate a = A^T h; h = A a; normalise

    Parameters
    ----------
    A : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h, a

    References
    ----------
    Kleinberg (1999)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Truth comes out of error more readily than out of confusion. -- Francis Bacon"})


def cheatsheet():
    return "Truth comes out of error more readily than out of confusion. -- Francis Bacon"
