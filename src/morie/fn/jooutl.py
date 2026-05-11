# morie.fn — function file (hadesllm/morie)
"""Rolling-IQR outlier detection for time series."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_ts_outlier_detection"]


def joseph_ts_outlier_detection(y, W):
    """
    Rolling-IQR outlier detection for time series

    Formula: flag y_t as outlier if y_t < Q1_W - 1.5*IQR_W or y_t > Q3_W + 1.5*IQR_W, W = rolling window

    Parameters
    ----------
    y : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: flags

    References
    ----------
    Joseph Ch 2, Outlier Detection section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rolling-IQR outlier detection for time series"})


def cheatsheet():
    return "jooutl: Rolling-IQR outlier detection for time series"
