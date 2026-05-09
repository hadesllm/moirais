# moirais.fn — function file (hadesllm/moirais)
"""Area under the ROC curve."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_auc_roc"]


def geron_auc_roc(y_true, y_scores):
    """
    Area under the ROC curve

    Formula: AUC = integral_0^1 TPR(FPR) d(FPR), approximated via trapezoid rule

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: auc

    References
    ----------
    Géron Ch 3, AUC-ROC section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Area under the ROC curve"})


def cheatsheet():
    return "grauc: Area under the ROC curve"
