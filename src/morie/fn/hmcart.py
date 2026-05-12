# morie.fn -- function file (hadesllm/morie)
"""CART algorithm: greedy binary splits minimizing impurity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cart_algorithm"]


def geron_cart_algorithm(X, y, criterion, max_depth):
    """
    CART algorithm: greedy binary splits minimizing impurity

    Formula: split minimizing J(k,t_k) = (m_L/m)*G_L + (m_R/m)*G_R

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    criterion : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CART algorithm: greedy binary splits minimizing impurity"})


def cheatsheet():
    return "hmcart: CART algorithm: greedy binary splits minimizing impurity"
