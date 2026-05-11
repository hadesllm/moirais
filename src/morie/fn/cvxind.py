"""Indicator function of set."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_indicator"]


def boyd_indicator(C, x):
    """
    Indicator function of set

    Formula: I_C(x) = 0 if x in C else +inf

    Parameters
    ----------
    C : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Indicator function of set"})


def cheatsheet():
    return "cvxind: Indicator function of set"
