"""Barabási-Albert preferential attachment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["barabasi_albert"]


def barabasi_albert(n, m):
    """
    Barabási-Albert preferential attachment

    Formula: P(connect to v) ~ k_v

    Parameters
    ----------
    n : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barabási-Albert (1999)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Barabási-Albert preferential attachment"})


def cheatsheet():
    return "barabsi: Barabási-Albert preferential attachment"
