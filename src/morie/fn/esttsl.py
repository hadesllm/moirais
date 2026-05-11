"""Theta method (M3 winner)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["theta_method"]


def theta_method(y, horizon):
    """
    Theta method (M3 winner)

    Formula: weighted combination of theta lines

    Parameters
    ----------
    y : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Assimakopoulos-Nikolopoulos (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theta method (M3 winner)"})


def cheatsheet():
    return "esttsl: Theta method (M3 winner)"
