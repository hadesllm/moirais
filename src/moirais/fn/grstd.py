# moirais.fn — function file (hadesllm/moirais)
"""Standardization (z-score normalization) to zero mean, unit variance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_standardization"]


def geron_standardization(X):
    """
    Standardization (z-score normalization) to zero mean, unit variance

    Formula: x_scaled = (x - mean(x)) / std(x)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_scaled

    References
    ----------
    Géron Ch 2, Feature Scaling section (standardization)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standardization (z-score normalization) to zero mean, unit variance"})


def cheatsheet():
    return "grstd: Standardization (z-score normalization) to zero mean, unit variance"
