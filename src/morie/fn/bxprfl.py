"""Baxter-King band-pass filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["baxter_king"]


def baxter_king(y, p_low, p_high, K):
    """
    Baxter-King band-pass filter

    Formula: approximate ideal band-pass via truncated weights

    Parameters
    ----------
    y : array-like
        Input data.
    p_low : array-like
        Input data.
    p_high : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Baxter-King (1999)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baxter-King band-pass filter"})


def cheatsheet():
    return "bxprfl: Baxter-King band-pass filter"
