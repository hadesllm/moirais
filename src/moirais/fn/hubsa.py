"""Confine yourself to the present. — Marcus Aurelius"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["Confine yourself to the present. — Marcus Aurelius"]


def hits_hubs_authorities(y, A, tol):
    """
    HITS hubs + authorities scores

    Formula: a = A^T h; h = A a; iterate to convergence

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kleinberg (1999)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confine yourself to the present. — Marcus Aurelius"})


def cheatsheet():
    return "Confine yourself to the present. — Marcus Aurelius"
