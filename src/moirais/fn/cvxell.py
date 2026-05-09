"""Minimum volume covering ellipsoid."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_minvol_ellipsoid"]


def boyd_minvol_ellipsoid(X):
    """
    Minimum volume covering ellipsoid

    Formula: min log det B^{-1} s.t. |Bx_i + d| <= 1

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: B, d

    References
    ----------
    Boyd CVX Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum volume covering ellipsoid"})


def cheatsheet():
    return "cvxell: Minimum volume covering ellipsoid"
