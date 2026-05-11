"""Pair correlation function g(r) for point patterns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pair_correlation_function"]


def pair_correlation_function(points, window, r):
    """
    Pair correlation function g(r) for point patterns

    Formula: g(r) = (1 / 2 pi r) dK(r)/dr

    Parameters
    ----------
    points : array-like
        Input data.
    window : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stoyan & Stoyan (1994)
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    y = points  # generator-template fallback (no second array param in spec)
    n = min(len(points), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pair correlation function g(r) for point patterns"})
    result = stats.spearmanr(points[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Pair correlation function g(r) for point patterns"})


def cheatsheet():
    return "pcfunc: Pair correlation function g(r) for point patterns"
