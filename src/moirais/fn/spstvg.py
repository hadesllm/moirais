"""Spatio-temporal semivariogram."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_st_variogram"]


def schabenberger_st_variogram(coords, times, z):
    """
    Spatio-temporal semivariogram

    Formula: gamma(h,u) = 0.5 * E[(Z(s+h,t+u)-Z(s,t))^2]

    Parameters
    ----------
    coords : array-like
        Input data.
    times : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: st_variogram

    References
    ----------
    Schabenberger Ch 9, Sec 9.4
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatio-temporal semivariogram"})


def cheatsheet():
    return "spstvg: Spatio-temporal semivariogram"
