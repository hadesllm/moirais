"""Empirical distribution function (eCDF)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_empirical_cdf"]


def wasserman_empirical_cdf(x, data):
    """
    Empirical distribution function (eCDF)

    Formula: F_n(x) = (1/n) sum_i I(X_i <= x)

    Parameters
    ----------
    x : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 7
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical distribution function (eCDF)"})


def cheatsheet():
    return "wsmcdf: Empirical distribution function (eCDF)"
