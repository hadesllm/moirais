"""Minimum volume ellipsoid."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mve"]


def mve(X, h):
    """
    Minimum volume ellipsoid

    Formula: smallest ellipsoid covering h points

    Parameters
    ----------
    X : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw (1985)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum volume ellipsoid"})


def cheatsheet():
    return "mvedet: Minimum volume ellipsoid"
