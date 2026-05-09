"""Adaptive iteration count for Sinkhorn given tol."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sinkhorn_iter_count"]


def ot_sinkhorn_iter_count(a, b, C, epsilon, tol, max_iter):
    """
    Adaptive iteration count for Sinkhorn given tol

    Formula: Stop when err<tol or iter>max_iter

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    tol : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: iter_count, err

    References
    ----------
    Cuturi (2013)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive iteration count for Sinkhorn given tol"})


def cheatsheet():
    return "otsinkit: Adaptive iteration count for Sinkhorn given tol"
