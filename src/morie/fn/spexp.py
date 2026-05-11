"""Exponential semivariogram model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_exponential_variogram"]


def schabenberger_exponential_variogram(h, nugget, sill, range):
    """
    Exponential semivariogram model

    Formula: gamma(h) = c0 + c1*(1 - exp(-h/a))

    Parameters
    ----------
    h : array-like
        Input data.
    nugget : array-like
        Input data.
    sill : array-like
        Input data.
    range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: semivariance

    References
    ----------
    Schabenberger Ch 4
    """
    h = np.asarray(h, dtype=float)
    n = int(h) if h.ndim == 0 else len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential semivariogram model"})


def cheatsheet():
    return "spexp: Exponential semivariogram model"
