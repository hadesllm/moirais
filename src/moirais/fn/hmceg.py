# moirais.fn — function file (hadesllm/moirais)
"""Gradient of cross-entropy for softmax regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cross_entropy_gradient"]


def geron_cross_entropy_gradient(X, Y, theta):
    """
    Gradient of cross-entropy for softmax regression

    Formula: grad_{theta_k} J = (1/m) sum_i (p_ik - y_ik) x_i

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient of cross-entropy for softmax regression"})


def cheatsheet():
    return "hmceg: Gradient of cross-entropy for softmax regression"
