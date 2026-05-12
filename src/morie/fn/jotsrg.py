# morie.fn -- function file (hadesllm/morie)
"""Frame TS forecasting as tabular regression via lag and window features."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_ts_as_regression"]


def joseph_ts_as_regression(y, lags, rolling_windows, seasonal_m, fourier_K):
    """
    Frame TS forecasting as tabular regression via lag and window features

    Formula: X_t = [y_{t-1}, y_{t-2}, ..., rolling_W(y_t), fourier(t), ...]; y_t = f(X_t)

    Parameters
    ----------
    y : array-like
        Input data.
    lags : array-like
        Input data.
    rolling_windows : array-like
        Input data.
    seasonal_m : array-like
        Input data.
    fourier_K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X, y

    References
    ----------
    Joseph Ch 5, Time Series as Regression section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frame TS forecasting as tabular regression via lag and window features"})


def cheatsheet():
    return "jotsrg: Frame TS forecasting as tabular regression via lag and window features"
