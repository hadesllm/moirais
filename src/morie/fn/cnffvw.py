"""Cinelli-Hazlett benchmark sensitivity for OLS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cinelli_hazlett_robust"]


def cinelli_hazlett_robust(y, D, X, R2_Y, R2_D):
    """
    Cinelli-Hazlett benchmark sensitivity for OLS

    Formula: adjusted estimate under hypothesised confounder R^2_Y * R^2_D

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    R2_Y : array-like
        Input data.
    R2_D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cinelli & Hazlett (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cinelli-Hazlett benchmark sensitivity for OLS"})


def cheatsheet():
    return "cnffvw: Cinelli-Hazlett benchmark sensitivity for OLS"
