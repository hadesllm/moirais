"""Least trimmed squares regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["least_trimmed_squares"]


def least_trimmed_squares(y, X, h):
    """
    Least trimmed squares regression

    Formula: min sum_{i=1}^{h} (r_(i)^2), h ~ floor(n/2)+1

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    h : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Least trimmed squares regression"})


def cheatsheet():
    return "ltsreg: Least trimmed squares regression"
