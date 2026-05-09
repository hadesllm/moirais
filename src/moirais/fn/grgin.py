# moirais.fn — function file (hadesllm/moirais)
"""Gini impurity at a tree node."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gini_impurity"]


def geron_gini_impurity(y):
    """
    Gini impurity at a tree node

    Formula: G_i = 1 - sum_{k=1..K} p_{i,k}^2

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gini

    References
    ----------
    Géron Ch 5, Eq 5-1 (Gini Impurity)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gini impurity at a tree node"})


def cheatsheet():
    return "grgin: Gini impurity at a tree node"
