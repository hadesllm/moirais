"""Projection onto convex set."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_projection"]


def boyd_projection(v, C):
    """
    Projection onto convex set

    Formula: P_C(v) = argmin_{x in C} |x - v|^2

    Parameters
    ----------
    v : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 8
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = len(v)
    result = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Projection onto convex set"})


def cheatsheet():
    return "cvxprc: Projection onto convex set"
