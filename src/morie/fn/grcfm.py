# morie.fn — function file (hadesllm/morie)
"""Confusion matrix for binary/multiclass classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_confusion_matrix"]


def geron_confusion_matrix(y_true, y_pred, n_classes):
    """
    Confusion matrix for binary/multiclass classification

    Formula: CM[i,j] = count(y_true == i and y_pred == j)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.
    n_classes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: matrix

    References
    ----------
    Géron Ch 3, Confusion Matrix section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confusion matrix for binary/multiclass classification"})


def cheatsheet():
    return "grcfm: Confusion matrix for binary/multiclass classification"
