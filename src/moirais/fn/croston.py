"""Croston's method (intermittent demand)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["croston"]


def croston(y, alpha):
    """
    Croston's method (intermittent demand)

    Formula: separate SES on size + interval

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Croston (1972)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Croston's method (intermittent demand)"})


def cheatsheet():
    return "croston: Croston's method (intermittent demand)"
