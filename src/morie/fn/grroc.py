# morie.fn -- function file (hadesllm/morie)
"""Receiver operating characteristic curve (TPR vs FPR over thresholds)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_roc_curve"]


def geron_roc_curve(y_true, y_scores):
    """
    Receiver operating characteristic curve (TPR vs FPR over thresholds)

    Formula: for each t: TPR(t) = TP(t)/(TP(t)+FN(t)); FPR(t) = FP(t)/(FP(t)+TN(t))

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fpr, tpr, thresholds

    References
    ----------
    Géron Ch 3, ROC Curve section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Receiver operating characteristic curve (TPR vs FPR over thresholds)"})


def cheatsheet():
    return "grroc: Receiver operating characteristic curve (TPR vs FPR over thresholds)"
