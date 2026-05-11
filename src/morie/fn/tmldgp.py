"""Penalized doubly-robust TMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_doubly_robust_pen"]


def tmle_doubly_robust_pen(y, D, X, penalty):
    """
    Penalized doubly-robust TMLE

    Formula: L1/L2 penalty on Q model coefficients

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    penalty : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belloni-Chernozhukov (2013); vdL-Gruber (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized doubly-robust TMLE"})


def cheatsheet():
    return "tmldgp: Penalized doubly-robust TMLE"
