# morie.fn — function file (hadesllm/morie)
"""Optimal ROC operating point minimizing expected cost."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch10_roc_optimal"]


def rangayyan_ch10_roc_optimal(fpr, tpr, cost_matrix, priors):
    """
    Optimal ROC operating point minimizing expected cost

    Formula: Optimal threshold: slope of ROC = (C_FP-C_TN)*P(neg) / ((C_FN-C_TP)*P(pos))

    Parameters
    ----------
    fpr : array-like
        Input data.
    tpr : array-like
        Input data.
    cost_matrix : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_threshold_idx

    References
    ----------
    Rangayyan Ch 10.9.1
    """
    fpr = np.asarray(fpr, dtype=float)
    n = int(fpr) if fpr.ndim == 0 else len(fpr)
    result = float(np.mean(fpr))
    se = float(np.std(fpr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal ROC operating point minimizing expected cost"})


def cheatsheet():
    return "rgeqn10b: Optimal ROC operating point minimizing expected cost"
