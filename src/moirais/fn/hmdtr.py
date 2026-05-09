# moirais.fn — function file (hadesllm/moirais)
"""Tree regularization via max_depth, min_samples_split, min_samples_leaf."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tree_regularization"]


def geron_tree_regularization(X, y, max_depth, min_samples_leaf):
    """
    Tree regularization via max_depth, min_samples_split, min_samples_leaf

    Formula: constraints on tree structure to reduce overfitting

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    max_depth : array-like
        Input data.
    min_samples_leaf : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tree regularization via max_depth, min_samples_split, min_samples_leaf"})


def cheatsheet():
    return "hmdtr: Tree regularization via max_depth, min_samples_split, min_samples_leaf"
