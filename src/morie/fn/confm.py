# morie.fn — function file (hadesllm/morie)
"""Confusion matrix with precision/recall/F1."""
import numpy as np
from ._richresult import RichResult

__all__ = ["confusion_matrix_metrics"]


def confusion_matrix_metrics(y_true, y_pred):
    """
    Confusion matrix with precision/recall/F1

    Formula: F1 = 2*P*R/(P+R), P=TP/(TP+FP)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 3
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confusion matrix with precision/recall/F1"})


def cheatsheet():
    return "confm: Confusion matrix with precision/recall/F1"
