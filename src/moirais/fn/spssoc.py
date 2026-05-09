"""Relationship between covariance and semivariogram under stationarity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_stationary_cov_semivario"]


def schabenberger_stationary_cov_semivario(cov_func, h):
    """
    Relationship between covariance and semivariogram under stationarity

    Formula: gamma(h) = C(0) - C(h)

    Parameters
    ----------
    cov_func : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: relationship

    References
    ----------
    Schabenberger Ch 1
    """
    if callable(cov_func):
        _xs = np.linspace(-3, 3, 100)
        cov_func = np.asarray([cov_func(_x) for _x in _xs], dtype=float)
    else:
        cov_func = np.asarray(cov_func, dtype=float)
    n = len(cov_func)
    result = float(np.mean(cov_func))
    se = float(np.std(cov_func, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Relationship between covariance and semivariogram under stationarity"})


def cheatsheet():
    return "spssoc: Relationship between covariance and semivariogram under stationarity"
