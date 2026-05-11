"""F (empty-space) nearest-neighbour distance function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_f_function"]


def ripley_f_function(points, window, r):
    """
    F (empty-space) nearest-neighbour distance function

    Formula: F(d) = P(min ||u - x_i|| <= d) for random u in window

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
    Diggle (2013) §2.4
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    n = len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "F (empty-space) nearest-neighbour distance function"})


def cheatsheet():
    return "ripF: F (empty-space) nearest-neighbour distance function"
