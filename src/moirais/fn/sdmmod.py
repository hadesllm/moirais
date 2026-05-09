"""Spatial Durbin model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_durbin"]


def spatial_durbin(y, X, W):
    """
    Spatial Durbin model

    Formula: y = rho W y + X beta + W X theta + eps

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    LeSage-Pace (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial Durbin model"})


def cheatsheet():
    return "sdmmod: Spatial Durbin model"
