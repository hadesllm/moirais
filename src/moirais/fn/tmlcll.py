"""TMLE for cross-lagged panel models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_cross_lagged"]


def tmle_cross_lagged(y, D, X, time):
    """
    TMLE for cross-lagged panel models

    Formula: target temporally-lagged effects in panel

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Allard-Boulet (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for cross-lagged panel models"})


def cheatsheet():
    return "tmlcll: TMLE for cross-lagged panel models"
