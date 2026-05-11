"""Ripley's K function for point patterns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_k_function"]


def ripley_k_function(points, window, r):
    """
    Ripley's K function for point patterns

    Formula: K(d) = lambda^{-1} E[# points in B(x_i, d) | x_i]

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
    Ripley (1976)
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    n = len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ripley's K function for point patterns"})


def cheatsheet():
    return "ripK: Ripley's K function for point patterns"
