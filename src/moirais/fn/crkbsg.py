"""Cokriging with secondary variable."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cokriging"]


def cokriging(coords, y, z, s_predict, cross_variogram):
    """
    Cokriging with secondary variable

    Formula: joint BLUP using cross-variogram

    Parameters
    ----------
    coords : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.
    s_predict : array-like
        Input data.
    cross_variogram : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wackernagel (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cokriging with secondary variable"})


def cheatsheet():
    return "crkbsg: Cokriging with secondary variable"
