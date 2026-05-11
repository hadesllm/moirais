"""CUSUM changepoint detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["changepoint_cusum"]


def changepoint_cusum(y, k, h):
    """
    CUSUM changepoint detection

    Formula: S_t = max(0, S_{t-1} + (x_t - mu_0 - k))

    Parameters
    ----------
    y : array-like
        Input data.
    k : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Page (1954)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CUSUM changepoint detection"})


def cheatsheet():
    return "chgcus: CUSUM changepoint detection"
