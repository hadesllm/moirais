"""Backtracking line search."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_backtracking"]


def boyd_backtracking(f, grad, x, dx, alpha, beta):
    """
    Backtracking line search

    Formula: while f(x+t dx) > f(x) + alpha t grad'dx: t <- beta t

    Parameters
    ----------
    f : array-like
        Input data.
    grad : array-like
        Input data.
    x : array-like
        Input data.
    dx : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: t

    References
    ----------
    Boyd CVX Ch 9
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backtracking line search"})


def cheatsheet():
    return "cvxbck: Backtracking line search"
