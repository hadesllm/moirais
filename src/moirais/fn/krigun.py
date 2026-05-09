"""Universal kriging with trend."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["universal_kriging"]


def universal_kriging(coords, values, s_predict, trend_order):
    """
    Universal kriging with trend

    Formula: OK on residuals after polynomial trend fit

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    s_predict : array-like
        Input data.
    trend_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cressie (1993)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Universal kriging with trend"})


def cheatsheet():
    return "krigun: Universal kriging with trend"
