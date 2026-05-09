# moirais.fn — function file (hadesllm/moirais)
"""Ridge (L2) regression cost."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ridge_cost"]


def geron_ridge_cost(X, y, theta, alpha):
    """
    Ridge (L2) regression cost

    Formula: J = MSE + alpha * (1/2) sum theta_i^2

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
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge (L2) regression cost"})


def cheatsheet():
    return "hmridg: Ridge (L2) regression cost"
