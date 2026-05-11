# morie.fn — function file (hadesllm/morie)
"""Classification tree leaf prediction (majority class)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tree_classification_leaf"]


def geron_tree_classification_leaf(y, leaf_mask):
    """
    Classification tree leaf prediction (majority class)

    Formula: y_hat_leaf = argmax_k sum_{i in leaf} 1{y_i == k}

    Parameters
    ----------
    y : array-like
        Input data.
    leaf_mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: leaf_class

    References
    ----------
    Géron Ch 5, Classification Trees section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classification tree leaf prediction (majority class)"})


def cheatsheet():
    return "grtrc: Classification tree leaf prediction (majority class)"
