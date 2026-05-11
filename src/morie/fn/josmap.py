# morie.fn — function file (hadesllm/morie)
"""Symmetric Mean Absolute Percentage Error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_smape"]


def joseph_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error

    Formula: sMAPE = (200/H) sum_h |y_h - y_hat_h| / (|y_h| + |y_hat_h|)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: smape

    References
    ----------
    Joseph Ch 19, sMAPE section
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetric Mean Absolute Percentage Error"})


def cheatsheet():
    return "josmap: Symmetric Mean Absolute Percentage Error"
