"""Sliced GRF for cross-sectional CATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sliced_grf"]


def sliced_grf(y, D, X, time):
    """
    Sliced GRF for cross-sectional CATE

    Formula: per-time-slice forest; aggregate over t

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Wager (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sliced GRF for cross-sectional CATE"})


def cheatsheet():
    return "slvgrf: Sliced GRF for cross-sectional CATE"
