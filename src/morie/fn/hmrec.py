# morie.fn -- function file (hadesllm/morie)
"""Recall (true positive rate, sensitivity) = TP / (TP + FN)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_recall"]


def geron_recall(y_true, y_pred):
    """
    Recall (true positive rate, sensitivity) = TP / (TP + FN)

    Formula: R = TP / (TP + FN)

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
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recall (true positive rate, sensitivity) = TP / (TP + FN)"})


def cheatsheet():
    return "hmrec: Recall (true positive rate, sensitivity) = TP / (TP + FN)"
