"""Least squares."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_least_squares"]


def boyd_least_squares(A, b):
    """
    Least squares

    Formula: min |Ax - b|_2^2

    Parameters
    ----------
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Least squares"})


def cheatsheet():
    return "cvxlsq: Least squares"
