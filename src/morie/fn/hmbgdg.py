# morie.fn -- function file (hadesllm/morie)
"""Gradient of linear-regression MSE cost."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_batch_gd_grad"]


def geron_batch_gd_grad(X, y, theta):
    """
    Gradient of linear-regression MSE cost

    Formula: grad J(theta) = (2/m) X^T (X theta - y)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient of linear-regression MSE cost"})


def cheatsheet():
    return "hmbgdg: Gradient of linear-regression MSE cost"
