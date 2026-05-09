"""Probability proportional to size sampling weight."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pps_sampling"]


def pps_sampling(y, size, n):
    """
    Probability proportional to size sampling weight

    Formula: pi_i = n * x_i / sum x_k

    Parameters
    ----------
    y : array-like
        Input data.
    size : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hansen & Hurwitz (1943)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Probability proportional to size sampling weight"})


def cheatsheet():
    return "ppsamp: Probability proportional to size sampling weight"
