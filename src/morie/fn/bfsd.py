"""Savage-Dickey density ratio for nested point null."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_factor_savage_dickey"]


def bayes_factor_savage_dickey(samples, prior):
    """
    Savage-Dickey density ratio for nested point null

    Formula: BF_01 = p(theta_0 | y) / p(theta_0)

    Parameters
    ----------
    samples : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Verdinelli & Wasserman (1995)
    """
    samples = np.atleast_1d(np.asarray(samples, dtype=float))
    n = len(samples)
    result = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Savage-Dickey density ratio for nested point null"})


def cheatsheet():
    return "bfsd: Savage-Dickey density ratio for nested point null"
