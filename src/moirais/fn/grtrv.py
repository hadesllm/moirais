# moirais.fn — function file (hadesllm/moirais)
"""Regression tree leaf prediction (leaf mean)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tree_regression_leaf"]


def geron_tree_regression_leaf(y, leaf_mask):
    """
    Regression tree leaf prediction (leaf mean)

    Formula: y_hat_leaf = (1/|leaf|) * sum_{i in leaf} y_i

    Parameters
    ----------
    y : array-like
        Input data.
    leaf_mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: leaf_mean

    References
    ----------
    Géron Ch 5, Regression Trees section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regression tree leaf prediction (leaf mean)"})


def cheatsheet():
    return "grtrv: Regression tree leaf prediction (leaf mean)"
