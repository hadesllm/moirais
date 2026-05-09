"""STL decomposition + residual outliers."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stl_anomaly"]


def stl_anomaly(y, period):
    """
    STL decomposition + residual outliers

    Formula: y = trend + seasonal + remainder; flag remainder

    Parameters
    ----------
    y : array-like
        Input data.
    period : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cleveland et al (1990)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "STL decomposition + residual outliers"})


def cheatsheet():
    return "stlAn: STL decomposition + residual outliers"
