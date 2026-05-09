# moirais.fn — function file (hadesllm/moirais)
"""Lasso (L1) regression cost function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lasso_cost"]


def geron_lasso_cost(X, y, theta, alpha):
    """
    Lasso (L1) regression cost function

    Formula: J(theta) = MSE(theta) + alpha * sum_{i=1..n} |theta_i|

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
    Géron Ch 4, Eq 4-10 (Lasso cost function)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lasso (L1) regression cost function"})


def cheatsheet():
    return "grlaso: Lasso (L1) regression cost function"
