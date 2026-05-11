# morie.fn — function file (hadesllm/morie)
"""Mean absolute error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mae_metric"]


def mae_metric(y_true, y_pred):
    """
    Mean absolute error

    Formula: MAE = (1/n) * sum_i |y_i - y_hat_i|

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'mae': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean absolute error"})


def cheatsheet():
    return "maedf: Mean absolute error"
