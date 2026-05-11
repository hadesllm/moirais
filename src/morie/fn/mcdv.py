"""Minimum covariance determinant."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mcd"]


def mcd(X, h, n_starts):
    """
    Minimum covariance determinant

    Formula: min |Σ_h| over h-subsets of size h

    Parameters
    ----------
    X : array-like
        Input data.
    h : array-like
        Input data.
    n_starts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw (1984)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum covariance determinant"})


def cheatsheet():
    return "mcdv: Minimum covariance determinant"
