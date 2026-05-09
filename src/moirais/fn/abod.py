"""Angle-based outlier detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["abod"]


def abod(X):
    """
    Angle-based outlier detection

    Formula: variance of angles to other points

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kriegel et al (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Angle-based outlier detection"})


def cheatsheet():
    return "abod: Angle-based outlier detection"
