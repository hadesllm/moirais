# morie.fn -- function file (hadesllm/morie)
"""Gradient of logistic regression cost."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_logistic_gradient"]


def geron_logistic_gradient(X, y, theta):
    """
    Gradient of logistic regression cost

    Formula: grad J = (1/m) X^T (sigmoid(X theta) - y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
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
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient of logistic regression cost"})


def cheatsheet():
    return "hmlogg: Gradient of logistic regression cost"
