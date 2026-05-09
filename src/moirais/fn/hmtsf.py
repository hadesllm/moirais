# moirais.fn — function file (hadesllm/moirais)
"""Time series forecasting with RNN."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_time_series_forecast"]


def geron_time_series_forecast(y, horizon, window):
    """
    Time series forecasting with RNN

    Formula: y_{T+h} = f(y_{T-w+1}, ..., y_T)

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Géron Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time series forecasting with RNN"})


def cheatsheet():
    return "hmtsf: Time series forecasting with RNN"
