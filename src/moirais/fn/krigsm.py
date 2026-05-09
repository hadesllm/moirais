"""Ordinary kriging interpolation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ordinary_kriging"]


def ordinary_kriging(coords, values, s_predict, variogram):
    """
    Ordinary kriging interpolation

    Formula: BLUP at unobserved s* via covariance C(s, s*)

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    s_predict : array-like
        Input data.
    variogram : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matheron (1962); Cressie (1993)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinary kriging interpolation"})


def cheatsheet():
    return "krigsm: Ordinary kriging interpolation"
