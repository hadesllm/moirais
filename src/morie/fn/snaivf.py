"""Seasonal naive ŷ_{t+h}=y_{t+h-m}."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["seasonal_naive"]


def seasonal_naive(y, m, h):
    """
    Seasonal naive ŷ_{t+h}=y_{t+h-m}

    Formula: copy from one period ago

    Parameters
    ----------
    y : array-like
        Input data.
    m : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hyndman-Athanasopoulos (2018) §3.1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Seasonal naive ŷ_{t+h}=y_{t+h-m}"})


def cheatsheet():
    return "snaivf: Seasonal naive ŷ_{t+h}=y_{t+h-m}"
