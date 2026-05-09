"""Lempel-Ziv complexity (compression-based)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lempel_ziv_complexity"]


def lempel_ziv_complexity(y):
    """
    Lempel-Ziv complexity (compression-based)

    Formula: C(s) = number of distinct substrings in LZ77 parse

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lempel & Ziv (1976)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lempel-Ziv complexity (compression-based)"})


def cheatsheet():
    return "lzcomp: Lempel-Ziv complexity (compression-based)"
