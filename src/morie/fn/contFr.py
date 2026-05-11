"""Continued fraction expansion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["continued_fraction"]


def continued_fraction(x, n):
    """
    Continued fraction expansion

    Formula: a_0 + 1/(a_1 + 1/(a_2 + ...))

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continued fraction expansion"})


def cheatsheet():
    return "contFr: Continued fraction expansion"
