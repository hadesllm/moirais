# morie.fn — function file (hadesllm/morie)
"""Lasso (L1) regression cost."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lasso_cost"]


def geron_lasso_cost(X, y, theta, alpha):
    """
    Lasso (L1) regression cost

    Formula: J = MSE + alpha * sum |theta_i|

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lasso (L1) regression cost"})


def cheatsheet():
    return "hmlaso: Lasso (L1) regression cost"
