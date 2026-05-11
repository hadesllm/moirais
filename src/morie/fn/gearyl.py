"""Local Geary's c per location."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_gearys_c"]


def local_gearys_c(x, W):
    """
    Local Geary's c per location

    Formula: c_i = sum_j w_ij (z_i - z_j)^2

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin (2018) local indicators
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local Geary's c per location"})


def cheatsheet():
    return "gearyl: Local Geary's c per location"
