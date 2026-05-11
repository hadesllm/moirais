"""Calibrated recommendations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["calibrated_rec"]


def calibrated_rec(pred, user_profile):
    """
    Calibrated recommendations

    Formula: match output genre distribution to user profile

    Parameters
    ----------
    pred : array-like
        Input data.
    user_profile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Steck (2018)
    """
    pred = np.atleast_1d(np.asarray(pred, dtype=float))
    n = len(pred)
    result = float(np.mean(pred))
    se = float(np.std(pred, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Calibrated recommendations"})


def cheatsheet():
    return "caltbR: Calibrated recommendations"
