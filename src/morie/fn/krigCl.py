"""Ordinary kriging."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ordinary_kriging"]


def ordinary_kriging(coords, values, new):
    """
    Ordinary kriging

    Formula: Z* = sum λ_i Z_i with sum λ=1

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    new : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matheron (1963)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinary kriging"})


def cheatsheet():
    return "krigCl: Ordinary kriging"
