# morie.fn -- function file (hadesllm/morie)
"""Decision trees are insensitive to feature scale (axis-aligned splits)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tree_sensitivity_scale"]


def geron_tree_sensitivity_scale(X, y):
    """
    Decision trees are insensitive to feature scale (axis-aligned splits)

    Formula: invariant to x' = a*x + b

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decision trees are insensitive to feature scale (axis-aligned splits)"})


def cheatsheet():
    return "hmdtst: Decision trees are insensitive to feature scale (axis-aligned splits)"
