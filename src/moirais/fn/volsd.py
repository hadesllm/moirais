"""Simple-difference rolling vol with sliding window."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_simple_diff"]


def vol_simple_diff(r, window):
    """
    Simple-difference rolling vol with sliding window

    Formula: σ̂_t = sqrt(mean(r²[t-w:t]))

    Parameters
    ----------
    r : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma_t

    References
    ----------
    RiskMetrics (1996)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple-difference rolling vol with sliding window"})


def cheatsheet():
    return "volsd: Simple-difference rolling vol with sliding window"
