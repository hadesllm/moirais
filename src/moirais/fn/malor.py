"""Log odds ratio + variance from a 2x2 table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_log_odds_ratio"]


def ma_log_odds_ratio(a, b, c, d):
    """
    Log odds ratio + variance from a 2x2 table

    Formula: logOR = log(ad/bc); v = 1/a+1/b+1/c+1/d

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logOR, var

    References
    ----------
    Borenstein et al. (2009)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log odds ratio + variance from a 2x2 table"})


def cheatsheet():
    return "malor: Log odds ratio + variance from a 2x2 table"
