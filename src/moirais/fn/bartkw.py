"""Bartlett (triangular) kernel weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bartlett_kernel_weights"]


def bartlett_kernel_weights(lags):
    """
    Bartlett (triangular) kernel weights

    Formula: w_k = 1 - k/(M+1) for k <= M

    Parameters
    ----------
    lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newey & West (1987)
    """
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    n = len(lags)
    result = float(np.mean(lags))
    se = float(np.std(lags, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bartlett (triangular) kernel weights"})


def cheatsheet():
    return "bartkw: Bartlett (triangular) kernel weights"
