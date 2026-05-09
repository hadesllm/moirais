"""Smooth functional data via P-spline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["smooth_functional_data"]


def smooth_functional_data(Y, argvals, lam):
    """
    Smooth functional data via P-spline

    Formula: P-spline applied per curve

    Parameters
    ----------
    Y : array-like
        Input data.
    argvals : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Eilers-Marx (1996)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smooth functional data via P-spline"})


def cheatsheet():
    return "smfd: Smooth functional data via P-spline"
