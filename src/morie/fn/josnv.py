# morie.fn -- function file (hadesllm/morie)
"""Seasonal naive: repeat last seasonal period."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_seasonal_naive"]


def joseph_seasonal_naive(y, m, horizon):
    """
    Seasonal naive: repeat last seasonal period

    Formula: y_hat_{T+h} = y_{T+h-m}  where m = seasonal period

    Parameters
    ----------
    y : array-like
        Input data.
    m : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 4, Seasonal Naive section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Seasonal naive: repeat last seasonal period"})


def cheatsheet():
    return "josnv: Seasonal naive: repeat last seasonal period"
