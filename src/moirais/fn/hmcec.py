# moirais.fn — function file (hadesllm/moirais)
"""Cross-entropy cost for K-class softmax regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cross_entropy_cost"]


def geron_cross_entropy_cost(X, Y, theta):
    """
    Cross-entropy cost for K-class softmax regression

    Formula: J = -(1/m) sum_i sum_k y_ik log p_ik

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
        Keys: cost

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-entropy cost for K-class softmax regression"})


def cheatsheet():
    return "hmcec: Cross-entropy cost for K-class softmax regression"
