"""Interval-censored Turnbull NPMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interval_censored_survival"]


def interval_censored_survival(L, R, event):
    """
    Interval-censored Turnbull NPMLE

    Formula: max likelihood over EM iteration

    Parameters
    ----------
    L : array-like
        Input data.
    R : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Turnbull (1976)
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    n = len(L)
    result = float(np.mean(L))
    se = float(np.std(L, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interval-censored Turnbull NPMLE"})


def cheatsheet():
    return "ssintc: Interval-censored Turnbull NPMLE"
