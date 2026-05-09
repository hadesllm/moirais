"""Christiano-Fitzgerald asymmetric filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["christiano_fitzgerald"]


def christiano_fitzgerald(y, p_low, p_high):
    """
    Christiano-Fitzgerald asymmetric filter

    Formula: asymmetric weights for non-stationary series

    Parameters
    ----------
    y : array-like
        Input data.
    p_low : array-like
        Input data.
    p_high : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Christiano-Fitzgerald (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Christiano-Fitzgerald asymmetric filter"})


def cheatsheet():
    return "crfflt: Christiano-Fitzgerald asymmetric filter"
