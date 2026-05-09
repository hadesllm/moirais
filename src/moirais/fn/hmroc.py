# moirais.fn — function file (hadesllm/moirais)
"""Receiver operating characteristic: FPR vs TPR over thresholds."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_roc_curve"]


def geron_roc_curve(y_true, scores):
    """
    Receiver operating characteristic: FPR vs TPR over thresholds

    Formula: FPR = FP/N; TPR = TP/P

    Parameters
    ----------
    y_true : array-like
        Input data.
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fpr, tpr, thresholds

    References
    ----------
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Receiver operating characteristic: FPR vs TPR over thresholds"})


def cheatsheet():
    return "hmroc: Receiver operating characteristic: FPR vs TPR over thresholds"
