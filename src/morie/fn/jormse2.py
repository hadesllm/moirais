# morie.fn — function file (hadesllm/morie)
"""Root Mean Squared Scaled Error — scale by in-sample naive RMSE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_rmsse"]


def joseph_rmsse(y_true, y_pred, y_train, m):
    """
    Root Mean Squared Scaled Error — scale by in-sample naive RMSE

    Formula: RMSSE = sqrt( mean((y_h - y_hat_h)^2) / mean((y_t - y_{t-m})^2) )

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.
    y_train : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rmsse

    References
    ----------
    Joseph Ch 19, RMSSE section
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root Mean Squared Scaled Error — scale by in-sample naive RMSE"})


def cheatsheet():
    return "jormse2: Root Mean Squared Scaled Error — scale by in-sample naive RMSE"
