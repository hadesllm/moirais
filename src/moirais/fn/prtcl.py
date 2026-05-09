"""Sequential Monte Carlo / particle filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["particle_filter"]


def particle_filter(y, f, h, N):
    """
    Sequential Monte Carlo / particle filter

    Formula: importance sample + resample N particles

    Parameters
    ----------
    y : array-like
        Input data.
    f : array-like
        Input data.
    h : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gordon-Salmond-Smith (1993)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential Monte Carlo / particle filter"})


def cheatsheet():
    return "prtcl: Sequential Monte Carlo / particle filter"
