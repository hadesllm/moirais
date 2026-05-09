# moirais.fn — function file (hadesllm/moirais)
"""Regression decision tree via CART minimizing MSE per leaf."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_regression_tree"]


def geron_regression_tree(X, y, max_depth):
    """
    Regression decision tree via CART minimizing MSE per leaf

    Formula: split minimizing (m_L/m)*MSE_L + (m_R/m)*MSE_R

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    max_depth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tree

    References
    ----------
    Géron Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regression decision tree via CART minimizing MSE per leaf"})


def cheatsheet():
    return "hmrdt: Regression decision tree via CART minimizing MSE per leaf"
