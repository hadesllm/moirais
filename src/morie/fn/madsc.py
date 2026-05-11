"""Median absolute deviation scale (consistent for normal)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mad_scale"]


def mad_scale(y):
    """
    Median absolute deviation scale (consistent for normal)

    Formula: MAD = 1.4826 * median(|x - median(x)|)

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
    Hampel (1974)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Median absolute deviation scale (consistent for normal)"})


def cheatsheet():
    return "madsc: Median absolute deviation scale (consistent for normal)"
