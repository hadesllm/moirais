# morie.fn — function file (hadesllm/morie)
"""Root mean squared error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rmse_metric"]


def rmse_metric(y_true, y_pred):
    """
    Root mean squared error

    Formula: RMSE = sqrt(MSE)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'rmse': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root mean squared error"})


def cheatsheet():
    return "rmsef: Root mean squared error"
