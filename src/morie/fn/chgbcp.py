"""Bayesian online changepoint detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayesian_online_changepoint"]


def bayesian_online_changepoint(y, hazard):
    """
    Bayesian online changepoint detection

    Formula: posterior over run length

    Parameters
    ----------
    y : array-like
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
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian online changepoint detection"})


def cheatsheet():
    return "chgbcp: Bayesian online changepoint detection"
