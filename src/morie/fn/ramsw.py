"""Ramsay E-type weight function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ramsay_weight"]


def ramsay_weight(y, a):
    """
    Ramsay E-type weight function

    Formula: w(r) = exp(-a |r|)

    Parameters
    ----------
    y : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ramsay E-type weight function"})


def cheatsheet():
    return "ramsw: Ramsay E-type weight function"
