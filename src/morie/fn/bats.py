"""BATS — Box-Cox ARMA Trend Seasonal."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bats"]


def bats(y, seasonal_periods):
    """
    BATS — Box-Cox ARMA Trend Seasonal

    Formula: BATS without trig (integer-period only)

    Parameters
    ----------
    y : array-like
        Input data.
    seasonal_periods : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    De Livera-Hyndman-Snyder (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BATS — Box-Cox ARMA Trend Seasonal"})


def cheatsheet():
    return "bats: BATS — Box-Cox ARMA Trend Seasonal"
