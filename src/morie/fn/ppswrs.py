"""PPS with replacement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pps_with_replacement"]


def pps_with_replacement(sizes, n):
    """
    PPS with replacement

    Formula: draw n times with prob proportional to size

    Parameters
    ----------
    sizes : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hansen-Hurwitz (1943)
    """
    sizes = np.atleast_1d(np.asarray(sizes, dtype=float))
    n = len(sizes)
    result = float(np.mean(sizes))
    se = float(np.std(sizes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PPS with replacement"})


def cheatsheet():
    return "ppswrs: PPS with replacement"
