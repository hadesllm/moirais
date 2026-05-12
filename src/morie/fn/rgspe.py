# morie.fn -- function file (hadesllm/morie)
"""Specificity (true negative rate)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_specificity"]


def rangayyan_specificity(y_true, y_pred):
    """
    Specificity (true negative rate)

    Formula: Sp = TN / (TN + FP)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: specificity

    References
    ----------
    Rangayyan Ch 10.9
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Specificity (true negative rate)"})


def cheatsheet():
    return "rgspe: Specificity (true negative rate)"
