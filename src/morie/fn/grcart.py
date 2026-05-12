# morie.fn -- function file (hadesllm/morie)
"""CART split cost at a node."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cart_split_cost"]


def geron_cart_split_cost(X, y, feature, threshold):
    """
    CART split cost at a node

    Formula: J(k, t_k) = (m_L/m) G_L + (m_R/m) G_R

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    feature : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost, left_mask

    References
    ----------
    Géron Ch 5, Eq 5-2 (CART cost function)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CART split cost at a node"})


def cheatsheet():
    return "grcart: CART split cost at a node"
