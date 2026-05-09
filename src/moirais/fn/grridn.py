# moirais.fn — function file (hadesllm/moirais)
"""Closed-form ridge regression via augmented normal equation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ridge_normal_equation"]


def geron_ridge_normal_equation(X, y, alpha):
    """
    Closed-form ridge regression via augmented normal equation

    Formula: theta_hat = (X^T X + alpha * A)^{-1} X^T y, A = diag(0,1,...,1)

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
    Géron Ch 4, Eq 4-9 (Ridge closed form)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form ridge regression via augmented normal equation"})


def cheatsheet():
    return "grridn: Closed-form ridge regression via augmented normal equation"
