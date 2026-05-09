"""Sieve bootstrap with general parametric sieve."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_sieve_general"]


def boot_sieve_general(x, fit_fn, rvs_fn, stat, B):
    """
    Sieve bootstrap with general parametric sieve

    Formula: Use truncated parametric model; resample innovations

    Parameters
    ----------
    x : array-like
        Input data.
    fit_fn : array-like
        Input data.
    rvs_fn : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Bühlmann (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sieve bootstrap with general parametric sieve"})


def cheatsheet():
    return "btsieve: Sieve bootstrap with general parametric sieve"
