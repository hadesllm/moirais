"""Log risk ratio + variance from a 2x2 table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_log_risk_ratio"]


def ma_log_risk_ratio(a, b, c, d):
    """
    Log risk ratio + variance from a 2x2 table

    Formula: logRR = log((a/(a+b))/(c/(c+d))); v=1/a-1/(a+b)+1/c-1/(c+d)

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
        Keys: logRR, var

    References
    ----------
    Borenstein et al. (2009)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log risk ratio + variance from a 2x2 table"})


def cheatsheet():
    return "malrr: Log risk ratio + variance from a 2x2 table"
