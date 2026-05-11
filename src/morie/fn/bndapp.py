"""Bound application example: returns to schooling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_application"]


def bound_application(y, D, X):
    """
    Bound application example: returns to schooling

    Formula: empirical application demonstrating method

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Pepper (2000) education
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound application example: returns to schooling"})


def cheatsheet():
    return "bndapp: Bound application example: returns to schooling"
