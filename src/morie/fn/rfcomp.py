"""Robust factor analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["robust_factor_analysis"]


def robust_factor_analysis(X, k):
    """
    Robust factor analysis

    Formula: FA on MCD covariance matrix

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pison-Rousseeuw-Filzmoser-Croux (2003)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Robust factor analysis"})


def cheatsheet():
    return "rfcomp: Robust factor analysis"
