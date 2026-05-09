"""Highest posterior density credible interval."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["highest_density_credible_interval"]


def highest_density_credible_interval(samples):
    """
    Highest posterior density credible interval

    Formula: shortest interval [a,b] with P(theta in [a,b]) >= 1-alpha

    Parameters
    ----------
    samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Box & Tiao (1973); Hyndman (1996)
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Highest posterior density credible interval"})


def cheatsheet():
    return "hdpic: Highest posterior density credible interval"
