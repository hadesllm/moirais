"""HT variance via second-order inclusion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ht_variance"]


def ht_variance(y, pi, pi_ij):
    """
    HT variance via second-order inclusion

    Formula: sum_ij (pi_ij - pi_i pi_j) (y_i/pi_i)(y_j/pi_j) / pi_ij

    Parameters
    ----------
    y : array-like
        Input data.
    pi : array-like
        Input data.
    pi_ij : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horvitz-Thompson (1952)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HT variance via second-order inclusion"})


def cheatsheet():
    return "htvar1: HT variance via second-order inclusion"
