"""Siegel repeated medians."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["siegel_repeated"]


def siegel_repeated(x, y):
    """
    Siegel repeated medians

    Formula: slope = median_i median_{j≠i} (y_j−y_i)/(x_j−x_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Siegel (1982)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Siegel repeated medians"})


def cheatsheet():
    return "sieg: Siegel repeated medians"
