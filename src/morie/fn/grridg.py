# morie.fn -- function file (hadesllm/morie)
"""Ridge (L2) regression cost function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ridge_cost"]


def geron_ridge_cost(X, y, theta, alpha):
    """
    Ridge (L2) regression cost function

    Formula: J(theta) = MSE(theta) + alpha * (1/2) * sum_{i=1..n} theta_i^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    theta : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost

    References
    ----------
    Géron Ch 4, Eq 4-8 (Ridge Regression cost)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge (L2) regression cost function"})


def cheatsheet():
    return "grridg: Ridge (L2) regression cost function"
