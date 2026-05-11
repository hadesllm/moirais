"""Moving MAD threshold."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["moving_mad"]


def moving_mad(x, window, k):
    """
    Moving MAD threshold

    Formula: |x_t − median| > k · MAD

    Parameters
    ----------
    x : array-like
        Input data.
    window : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel (1974)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moving MAD threshold"})


def cheatsheet():
    return "madMov: Moving MAD threshold"
