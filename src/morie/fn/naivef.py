"""Naive ŷ_{t+h}=y_t."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["naive_forecast"]


def naive_forecast(y, h):
    """
    Naive ŷ_{t+h}=y_t

    Formula: y_T for all horizons

    Parameters
    ----------
    y : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Naive ŷ_{t+h}=y_t"})


def cheatsheet():
    return "naivef: Naive ŷ_{t+h}=y_t"
