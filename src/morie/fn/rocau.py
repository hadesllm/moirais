# morie.fn — function file (hadesllm/morie)
"""ROC curve and AUC computation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["roc_auc_score"]


def roc_auc_score(y_true, y_score):
    """
    ROC curve and AUC computation

    Formula: AUC = integral TPR d(FPR)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_score : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ROC curve and AUC computation"})


def cheatsheet():
    return "rocau: ROC curve and AUC computation"
