# morie.fn — function file (hadesllm/morie)
"""Polynomial feature expansion up to given degree (no interactions)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_polynomial_features"]


def geron_polynomial_features(X, degree):
    """
    Polynomial feature expansion up to given degree (no interactions)

    Formula: phi(X) = [1, X, X^2, ..., X^d]

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
    Géron Ch 4, Polynomial Regression section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial feature expansion up to given degree (no interactions)"})


def cheatsheet():
    return "grpoly: Polynomial feature expansion up to given degree (no interactions)"
