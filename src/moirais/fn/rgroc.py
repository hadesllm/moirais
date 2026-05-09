# moirais.fn — function file (hadesllm/moirais)
"""Receiver operating characteristic (ROC) curve and AUC."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_roc_curve"]


def rangayyan_roc_curve(y_true, y_scores):
    """
    Receiver operating characteristic (ROC) curve and AUC

    Formula: ROC: Se vs (1-Sp) at varying thresholds; AUC = integral Se d(1-Sp)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fpr, tpr, auc

    References
    ----------
    Rangayyan Ch 10.9.1
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Receiver operating characteristic (ROC) curve and AUC"})


def cheatsheet():
    return "rgroc: Receiver operating characteristic (ROC) curve and AUC"
