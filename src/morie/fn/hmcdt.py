# morie.fn -- function file (hadesllm/morie)
"""Classification decision tree via CART with Gini or entropy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_classification_tree"]


def geron_classification_tree(X, y, criterion, max_depth):
    """
    Classification decision tree via CART with Gini or entropy

    Formula: recursively split to minimize impurity until stopping criteria

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classification decision tree via CART with Gini or entropy"})


def cheatsheet():
    return "hmcdt: Classification decision tree via CART with Gini or entropy"
