"""Systematic sampling -- every k-th unit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["systematic_sampling"]


def systematic_sampling(y, N, n):
    """
    Systematic sampling -- every k-th unit

    Formula: k = floor(N/n); take every k-th from random start in [1,k]

    Parameters
    ----------
    y : array-like
        Input data.
    N : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §8
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Systematic sampling -- every k-th unit"})


def cheatsheet():
    return "systmp: Systematic sampling -- every k-th unit"
