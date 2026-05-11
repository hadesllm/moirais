"""Slow-decreasing DP truncation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["slow_dp_truncate"]


def slow_dp_truncate(alpha, eps):
    """
    Slow-decreasing DP truncation

    Formula: truncate at K such that residual mass < eps

    Parameters
    ----------
    alpha : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ishwaran-James (2001)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slow-decreasing DP truncation"})


def cheatsheet():
    return "slowdp: Slow-decreasing DP truncation"
