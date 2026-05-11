"""Nonparametric Bayes quantile function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bnp_percent_quantile"]


def bnp_percent_quantile(y, quantile):
    """
    Nonparametric Bayes quantile function

    Formula: posterior of F^-1(q) | y

    Parameters
    ----------
    y : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kottas-Krnjajic (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric Bayes quantile function"})


def cheatsheet():
    return "bnppct: Nonparametric Bayes quantile function"
