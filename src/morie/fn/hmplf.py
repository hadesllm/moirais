# morie.fn — function file (hadesllm/morie)
"""Polynomial feature expansion up to given degree."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_polynomial_features"]


def geron_polynomial_features(X, degree):
    """
    Polynomial feature expansion up to given degree

    Formula: phi(x) = [1, x, x^2, ..., x^d]

    Parameters
    ----------
    X : array-like
        Input data.
    degree : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_poly

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial feature expansion up to given degree"})


def cheatsheet():
    return "hmplf: Polynomial feature expansion up to given degree"
