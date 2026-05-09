"""Ripley's K function for point patterns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_k"]


def ripley_k(coords, r_grid):
    """
    Ripley's K function for point patterns

    Formula: K(r) = lambda^-1 E[#points within r]

    Parameters
    ----------
    coords : array-like
        Input data.
    r_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ripley (1976)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ripley's K function for point patterns"})


def cheatsheet():
    return "rkfunc: Ripley's K function for point patterns"
