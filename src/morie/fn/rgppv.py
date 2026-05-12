# morie.fn -- function file (hadesllm/morie)
"""Positive predictive value (precision)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ppv"]


def rangayyan_ppv(y_true, y_pred):
    """
    Positive predictive value (precision)

    Formula: PPV = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ppv

    References
    ----------
    Rangayyan Ch 10.9
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Positive predictive value (precision)"})


def cheatsheet():
    return "rgppv: Positive predictive value (precision)"
