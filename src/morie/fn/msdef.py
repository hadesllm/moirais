# morie.fn — function file (hadesllm/morie)
"""Mean squared error prediction metric."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mse_metric"]


def mse_metric(y_true, y_pred):
    """
    Mean squared error prediction metric

    Formula: MSE = (1/n) * sum_i (y_i - y_hat_i)^2

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'mse': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean squared error prediction metric"})


def cheatsheet():
    return "msdef: Mean squared error prediction metric"
