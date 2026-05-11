"""Lottery sampling (simple random sample without replacement)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lottery_sampling"]


def lottery_sampling(y, N, n):
    """
    Lottery sampling (simple random sample without replacement)

    Formula: each subset of size n equally likely; pi_i = n/N

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
    Cochran (1977) §2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lottery sampling (simple random sample without replacement)"})


def cheatsheet():
    return "lottosi: Lottery sampling (simple random sample without replacement)"
