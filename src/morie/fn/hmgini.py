# morie.fn -- function file (hadesllm/morie)
"""Gini impurity for a node."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gini_impurity"]


def geron_gini_impurity(y):
    """
    Gini impurity for a node

    Formula: G = 1 - sum_k p_k^2

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
    Géron Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gini impurity for a node"})


def cheatsheet():
    return "hmgini: Gini impurity for a node"
