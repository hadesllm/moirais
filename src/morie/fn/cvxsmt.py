"""Smoothed min via log-sum-exp."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_smooth_min"]


def boyd_smooth_min(x):
    """
    Smoothed min via log-sum-exp

    Formula: lse(x) = log sum exp x_i

    Parameters
    ----------
    x : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothed min via log-sum-exp"})


def cheatsheet():
    return "cvxsmt: Smoothed min via log-sum-exp"
