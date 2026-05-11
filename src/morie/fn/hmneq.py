# morie.fn — function file (hadesllm/morie)
"""Closed-form OLS via the normal equation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_normal_equation"]


def geron_normal_equation(X, y):
    """
    Closed-form OLS via the normal equation

    Formula: theta_hat = (X^T X)^{-1} X^T y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form OLS via the normal equation"})


def cheatsheet():
    return "hmneq: Closed-form OLS via the normal equation"
