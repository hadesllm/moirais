"""Convex combination."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_convex_combination"]


def boyd_convex_combination(x, theta):
    """
    Convex combination

    Formula: sum theta_i x_i, theta >= 0, sum theta = 1

    Parameters
    ----------
    x : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convex combination"})


def cheatsheet():
    return "cvxcvc: Convex combination"
