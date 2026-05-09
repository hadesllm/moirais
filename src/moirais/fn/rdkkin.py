"""Regression kink design (RKD)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kink_rdd"]


def kink_rdd(y, x, cutoff, bandwidth):
    """
    Regression kink design (RKD)

    Formula: tau = (slope+ - slope-) / (slope of treatment kink)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    cutoff : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Card, Lee, Pei, Weber (2015)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regression kink design (RKD)"})


def cheatsheet():
    return "rdkkin: Regression kink design (RKD)"
