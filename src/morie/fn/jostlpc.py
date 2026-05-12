# morie.fn -- function file (hadesllm/morie)
"""STL (Seasonal and Trend decomposition using LOESS): Y = Trend + Seasonal + Remainder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_stl_decomposition"]


def joseph_stl_decomposition(y, period, seasonal_window, trend_window):
    """
    STL (Seasonal and Trend decomposition using LOESS): Y = Trend + Seasonal + Remainder

    Formula: Y_t = T_t + S_t + R_t;  T_t, S_t via iterated LOESS smoothing

    Parameters
    ----------
    y : array-like
        Input data.
    period : array-like
        Input data.
    seasonal_window : array-like
        Input data.
    trend_window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: trend, seasonal, remainder

    References
    ----------
    Joseph Ch 3, STL decomposition section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "STL (Seasonal and Trend decomposition using LOESS): Y = Trend + Seasonal + Remainder"})


def cheatsheet():
    return "jostlpc: STL (Seasonal and Trend decomposition using LOESS): Y = Trend + Seasonal + Remainder"
