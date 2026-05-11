# morie.fn — function file (hadesllm/morie)
"""Naive forecast: last observed value carried forward."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_naive_forecast"]


def joseph_naive_forecast(y, horizon):
    """
    Naive forecast: last observed value carried forward

    Formula: y_hat_{T+h} = y_T  for all h >= 1

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 4, Naive Forecast baseline
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Naive forecast: last observed value carried forward"})


def cheatsheet():
    return "jonaiv: Naive forecast: last observed value carried forward"
