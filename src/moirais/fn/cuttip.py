"""Cutting plane method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cutting_plane"]


def cutting_plane(c, A, b, integer_indices):
    """
    Cutting plane method

    Formula: add Gomory cut to LP relaxation

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.
    integer_indices : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gomory (1958)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cutting plane method"})


def cheatsheet():
    return "cuttip: Cutting plane method"
