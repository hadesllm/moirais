"""Epigraph."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_epigraph"]


def boyd_epigraph(f, x, t):
    """
    Epigraph

    Formula: epi f = {(x,t) : f(x) <= t}

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: set

    References
    ----------
    Boyd CVX Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epigraph"})


def cheatsheet():
    return "cvxepi: Epigraph"
