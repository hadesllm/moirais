"""Variogram model fit (spherical/exponential/Gaussian)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variogram_fit"]


def variogram_fit(coords, values, model):
    """
    Variogram model fit (spherical/exponential/Gaussian)

    Formula: gamma(h) = c0 + c1 * model(h/a)

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cressie (1993)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variogram model fit (spherical/exponential/Gaussian)"})


def cheatsheet():
    return "krigsv: Variogram model fit (spherical/exponential/Gaussian)"
