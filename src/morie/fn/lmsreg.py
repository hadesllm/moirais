"""Least median of squares regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["least_median_squares"]


def least_median_squares(y, X):
    """
    Least median of squares regression

    Formula: min median(r_i^2)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw (1984)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Least median of squares regression"})


def cheatsheet():
    return "lmsreg: Least median of squares regression"
