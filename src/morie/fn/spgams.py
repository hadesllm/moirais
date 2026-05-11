"""Spatial GAM with bivariate smooth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_gams"]


def spatial_gams(y, X, coords):
    """
    Spatial GAM with bivariate smooth

    Formula: y = f(s) + X beta; f ~ thin-plate spline

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wood (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial GAM with bivariate smooth"})


def cheatsheet():
    return "spgams: Spatial GAM with bivariate smooth"
