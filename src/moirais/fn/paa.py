"""Piecewise aggregate approximation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["paa"]


def paa(x, N):
    """
    Piecewise aggregate approximation

    Formula: split into N segments, average each

    Parameters
    ----------
    x : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Keogh et al (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Piecewise aggregate approximation"})


def cheatsheet():
    return "paa: Piecewise aggregate approximation"
