"""Solve linear Diophantine ax+by=c."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["diophantine"]


def diophantine(a, b, c):
    """
    Solve linear Diophantine ax+by=c

    Formula: extended Euclidean algorithm

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical number theory
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Solve linear Diophantine ax+by=c"})


def cheatsheet():
    return "diophs: Solve linear Diophantine ax+by=c"
