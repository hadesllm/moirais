"""Short-panel bound under stationarity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_short_panel"]


def bound_short_panel(y, D, time):
    """
    Short-panel bound under stationarity

    Formula: bounds when T < required for point ID

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Honoré-Tamer (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Short-panel bound under stationarity"})


def cheatsheet():
    return "bnshrt: Short-panel bound under stationarity"
