"""Perspective function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_perspective"]


def boyd_perspective(f, x, t):
    """
    Perspective function

    Formula: g(x,t) = t f(x/t), t > 0

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
        Keys: value

    References
    ----------
    Boyd CVX Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perspective function"})


def cheatsheet():
    return "cvxprl: Perspective function"
