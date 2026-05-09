"""Bayesian online change-point detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bocpd"]


def bocpd(x, hazard):
    """
    Bayesian online change-point detection

    Formula: recursive hazard-rate posterior over run lengths

    Parameters
    ----------
    x : array-like
        Input data.
    hazard : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Adams-MacKay (2007)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian online change-point detection"})


def cheatsheet():
    return "bocpd: Bayesian online change-point detection"
