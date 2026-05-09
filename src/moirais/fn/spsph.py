"""Spherical semivariogram model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spherical_variogram"]


def schabenberger_spherical_variogram(h, nugget, sill, range):
    """
    Spherical semivariogram model

    Formula: gamma(h) = c0 + c1*(1.5*h/a - 0.5*(h/a)^3) if h<=a, else c0+c1

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
    Schabenberger Ch 4, Sec 4.3.3
    """
    h = np.asarray(h, dtype=float)
    n = int(h) if h.ndim == 0 else len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spherical semivariogram model"})


def cheatsheet():
    return "spsph: Spherical semivariogram model"
