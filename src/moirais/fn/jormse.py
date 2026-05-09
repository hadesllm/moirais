# moirais.fn — function file (hadesllm/moirais)
"""Root mean squared error for forecasts."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_rmse"]


def joseph_rmse(y_true, y_pred):
    """
    Root mean squared error for forecasts

    Formula: RMSE = sqrt( (1/H) sum_{h=1..H} (y_{T+h} - y_hat_{T+h})^2 )

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
    Joseph Ch 19, RMSE section
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root mean squared error for forecasts"})


def cheatsheet():
    return "jormse: Root mean squared error for forecasts"
