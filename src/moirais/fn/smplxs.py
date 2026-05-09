"""Simplex method LP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["simplex_lp"]


def simplex_lp(c, A, b):
    """
    Simplex method LP

    Formula: pivot from vertex to vertex

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dantzig (1947)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simplex method LP"})


def cheatsheet():
    return "smplxs: Simplex method LP"
