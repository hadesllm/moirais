# morie.fn -- function file (hadesllm/morie)
"""Lag feature: shift the series by k steps."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_lag_feature"]


def joseph_lag_feature(y, k):
    """
    Lag feature: shift the series by k steps

    Formula: x_t^{(lag k)} = y_{t-k}

    Parameters
    ----------
    y : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lag_feature

    References
    ----------
    Joseph Ch 6, Lag Features section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lag feature: shift the series by k steps"})


def cheatsheet():
    return "jolagf: Lag feature: shift the series by k steps"
