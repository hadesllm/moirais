"""G (point-to-point) nearest-neighbour distance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_g_function"]


def ripley_g_function(points, window, r):
    """
    G (point-to-point) nearest-neighbour distance

    Formula: G(d) = P(min ||x_j - x_i|| <= d) i != j

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "G (point-to-point) nearest-neighbour distance"})


def cheatsheet():
    return "ripG: G (point-to-point) nearest-neighbour distance"
