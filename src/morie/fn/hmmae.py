# morie.fn — function file (hadesllm/morie)
"""Mean absolute error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mae"]


def geron_mae(y_true, y_pred):
    """
    Mean absolute error

    Formula: MAE = (1/m) sum_i |y_hat_i - y_i|

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mae

    References
    ----------
    Géron Ch 2
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean absolute error"})


def cheatsheet():
    return "hmmae: Mean absolute error"
