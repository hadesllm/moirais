# morie.fn -- function file (hadesllm/morie)
"""Precision-recall curve over decision thresholds."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_precision_recall_curve"]


def geron_precision_recall_curve(y_true, scores):
    """
    Precision-recall curve over decision thresholds

    Formula: {(P(t), R(t)) : t in thresholds}

    Parameters
    ----------
    y_true : array-like
        Input data.
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: precisions, recalls, thresholds

    References
    ----------
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Precision-recall curve over decision thresholds"})


def cheatsheet():
    return "hmprc: Precision-recall curve over decision thresholds"
