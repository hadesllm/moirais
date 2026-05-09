# moirais.fn — function file (hadesllm/moirais)
"""Standardization (z-score): zero mean, unit variance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_standardization"]


def geron_standardization(X):
    """
    Standardization (z-score): zero mean, unit variance

    Formula: x' = (x - mu) / sigma

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_std

    References
    ----------
    Géron Ch 2
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standardization (z-score): zero mean, unit variance"})


def cheatsheet():
    return "hmstz: Standardization (z-score): zero mean, unit variance"
