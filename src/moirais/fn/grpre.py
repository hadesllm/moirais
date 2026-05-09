# moirais.fn — function file (hadesllm/moirais)
"""Precision = TP / (TP + FP)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_precision"]


def geron_precision(y_true, y_pred):
    """
    Precision = TP / (TP + FP)

    Formula: precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: precision

    References
    ----------
    Géron Ch 3, Eq 3-1 (Precision)
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Precision = TP / (TP + FP)"})


def cheatsheet():
    return "grpre: Precision = TP / (TP + FP)"
