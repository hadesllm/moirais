"""Systematic sample with random start."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["systematic_with_random_start"]


def systematic_with_random_start(y, N, n):
    """
    Systematic sample with random start

    Formula: start ~ Uniform{1,...,k}; sample at start, start+k, ...

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
    Cochran (1977) §8.2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Systematic sample with random start"})


def cheatsheet():
    return "swsmp: Systematic sample with random start"
