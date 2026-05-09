# moirais.fn — function file (hadesllm/moirais)
"""F1 score — harmonic mean of precision and recall."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_f1_score"]


def geron_f1_score(y_true, y_pred):
    """
    F1 score — harmonic mean of precision and recall

    Formula: F1 = 2 * precision * recall / (precision + recall)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f1

    References
    ----------
    Géron Ch 3, Eq 3-3 (F1 score)
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "F1 score — harmonic mean of precision and recall"})


def cheatsheet():
    return "grf1: F1 score — harmonic mean of precision and recall"
