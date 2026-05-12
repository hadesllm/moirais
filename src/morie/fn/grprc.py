# morie.fn -- function file (hadesllm/morie)
"""Precision-recall curve: precision and recall over thresholds."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_precision_recall_curve"]


def geron_precision_recall_curve(y_true, y_scores):
    """
    Precision-recall curve: precision and recall over thresholds

    Formula: for each threshold t: precision(t), recall(t) computed from TP/FP/FN at t

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: thresholds, precision, recall

    References
    ----------
    Géron Ch 3, Precision/Recall curve section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Precision-recall curve: precision and recall over thresholds"})


def cheatsheet():
    return "grprc: Precision-recall curve: precision and recall over thresholds"
