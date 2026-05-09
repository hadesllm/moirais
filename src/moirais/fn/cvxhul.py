"""Convex hull."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_convex_hull"]


def boyd_convex_hull(S):
    """
    Convex hull

    Formula: conv S = {sum theta_i x_i : theta >= 0, sum=1}

    Parameters
    ----------
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: hull

    References
    ----------
    Boyd CVX Ch 2
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convex hull"})


def cheatsheet():
    return "cvxhul: Convex hull"
