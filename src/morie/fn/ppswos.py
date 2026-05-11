"""PPS without replacement (Sampford)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pps_without_replacement"]


def pps_without_replacement(sizes, n):
    """
    PPS without replacement (Sampford)

    Formula: sequential PPS without replacement

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
    Sampford (1967)
    """
    sizes = np.atleast_1d(np.asarray(sizes, dtype=float))
    n = len(sizes)
    result = float(np.mean(sizes))
    se = float(np.std(sizes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PPS without replacement (Sampford)"})


def cheatsheet():
    return "ppswos: PPS without replacement (Sampford)"
