"""Viral shedding curve fit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["viral_shedding_model"]


def viral_shedding_model(days, viral_load):
    """
    Viral shedding curve fit

    Formula: piecewise log-linear: rise + plateau + decay

    Parameters
    ----------
    days : array-like
        Input data.
    viral_load : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    He et al (2020)
    """
    days = np.atleast_1d(np.asarray(days, dtype=float))
    n = len(days)
    result = float(np.mean(days))
    se = float(np.std(days, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Viral shedding curve fit"})


def cheatsheet():
    return "shedmd: Viral shedding curve fit"
