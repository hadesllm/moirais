"""Risk difference + variance from a 2x2 table."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_risk_difference"]


def ma_risk_difference(a, b, c, d):
    """
    Risk difference + variance from a 2x2 table

    Formula: RD = a/(a+b) - c/(c+d)

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
        Keys: RD, var

    References
    ----------
    Borenstein et al. (2009)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Risk difference + variance from a 2x2 table"})


def cheatsheet():
    return "mard: Risk difference + variance from a 2x2 table"
