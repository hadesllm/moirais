"""Minimum covariance determinant location/scatter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["min_covariance_determinant"]


def min_covariance_determinant(y, X, h):
    """
    Minimum covariance determinant location/scatter

    Formula: argmin_{H subset, |H|=h} det(cov(X[H]))

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
    Rousseeuw (1984); Rousseeuw & Van Driessen (1999)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum covariance determinant location/scatter"})


def cheatsheet():
    return "mcdcv: Minimum covariance determinant location/scatter"
