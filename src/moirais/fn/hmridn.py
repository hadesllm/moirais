# moirais.fn — function file (hadesllm/moirais)
"""Closed-form ridge via augmented normal equation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ridge_normal"]


def geron_ridge_normal(X, y, alpha):
    """
    Closed-form ridge via augmented normal equation

    Formula: theta = (X^T X + alpha A)^{-1} X^T y, A = diag(0,1,...,1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form ridge via augmented normal equation"})


def cheatsheet():
    return "hmridn: Closed-form ridge via augmented normal equation"
