# moirais.fn — function file (hadesllm/moirais)
"""Area under the ROC curve."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_auc_roc"]


def geron_auc_roc(y_true, scores):
    """
    Area under the ROC curve

    Formula: AUC = integral_0^1 TPR(FPR) dFPR

    Parameters
    ----------
    y_true : array-like
        Input data.
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: auc

    References
    ----------
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Area under the ROC curve"})


def cheatsheet():
    return "hmauc: Area under the ROC curve"
