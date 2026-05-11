"""Proportional stratum allocation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["proportional_allocation"]


def proportional_allocation(N, Nh, n):
    """
    Proportional stratum allocation

    Formula: n_h = n N_h / N

    Parameters
    ----------
    N : array-like
        Input data.
    Nh : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neyman (1934)
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportional stratum allocation"})


def cheatsheet():
    return "propal: Proportional stratum allocation"
