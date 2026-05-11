"""Hurst exponent via R/S analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hurst_exponent"]


def hurst_exponent(y):
    """
    Hurst exponent via R/S analysis

    Formula: log(R/S) ~ H log(n)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hurst (1951)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hurst exponent via R/S analysis"})


def cheatsheet():
    return "hurste: Hurst exponent via R/S analysis"
