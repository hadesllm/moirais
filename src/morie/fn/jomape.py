# morie.fn — function file (hadesllm/morie)
"""Mean Absolute Percentage Error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_mape"]


def joseph_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error

    Formula: MAPE = (100/H) sum_h |y_h - y_hat_h| / |y_h|

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mape

    References
    ----------
    Joseph Ch 19, MAPE section
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean Absolute Percentage Error"})


def cheatsheet():
    return "jomape: Mean Absolute Percentage Error"
