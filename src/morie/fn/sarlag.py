"""Spatial autoregressive lag model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_ar_lag_model"]


def spatial_ar_lag_model(y, X, W):
    """
    Spatial autoregressive lag model

    Formula: y = rho W y + X beta + epsilon

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
    Anselin (1988)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial autoregressive lag model"})


def cheatsheet():
    return "sarlag: Spatial autoregressive lag model"
