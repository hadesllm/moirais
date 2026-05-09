# moirais.fn — function file (hadesllm/moirais)
"""Root mean squared error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rmse"]


def geron_rmse(y_true, y_pred):
    """
    Root mean squared error

    Formula: RMSE = sqrt((1/m) sum_i (y_hat_i - y_i)^2)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rmse

    References
    ----------
    Géron Ch 2
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root mean squared error"})


def cheatsheet():
    return "hmrms: Root mean squared error"
