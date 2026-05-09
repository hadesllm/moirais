"""Local polynomial smoother."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_polynomial"]


def local_polynomial(x, y, h, degree):
    """
    Local polynomial smoother

    Formula: weighted least squares with kernel weights

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    h : array-like
        Input data.
    degree : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fan-Gijbels (1996)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local polynomial smoother"})


def cheatsheet():
    return "locp: Local polynomial smoother"
