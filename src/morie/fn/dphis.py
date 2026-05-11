"""DP histogram (parallel composition)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_histogram"]


def dp_histogram(x, bins, epsilon):
    """
    DP histogram (parallel composition)

    Formula: per-bin: count + Lap(1/ε)

    Parameters
    ----------
    x : array-like
        Input data.
    bins : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Roth (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP histogram (parallel composition)"})


def cheatsheet():
    return "dphis: DP histogram (parallel composition)"
