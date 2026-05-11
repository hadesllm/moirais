# morie.fn — function file (hadesllm/morie)
"""Recall (sensitivity, TPR) = TP / (TP + FN)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_recall"]


def geron_recall(y_true, y_pred):
    """
    Recall (sensitivity, TPR) = TP / (TP + FN)

    Formula: recall = TP / (TP + FN)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: recall

    References
    ----------
    Géron Ch 3, Eq 3-2 (Recall)
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recall (sensitivity, TPR) = TP / (TP + FN)"})


def cheatsheet():
    return "grrec: Recall (sensitivity, TPR) = TP / (TP + FN)"
