"""Linear detrending."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["detrend_climate"]


def detrend_climate(x, t):
    """
    Linear detrending

    Formula: x_t − (a + b t)

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    standard
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear detrending"})


def cheatsheet():
    return "detrnd: Linear detrending"
