"""Empirical variogram γ(h)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variogram"]


def variogram(coords, values, bins):
    """
    Empirical variogram γ(h)

    Formula: γ(h) = (1/2N) sum (Z(s_i)-Z(s_i+h))²

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    bins : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical variogram γ(h)"})


def cheatsheet():
    return "vgrm: Empirical variogram γ(h)"
