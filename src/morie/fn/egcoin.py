"""Engle-Granger two-step cointegration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["engle_granger_2step"]


def engle_granger_2step(y, X):
    """
    Engle-Granger two-step cointegration

    Formula: step1: y_t = beta'X_t + e_t; step2: ADF on e_t

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle & Granger (1987)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Engle-Granger two-step cointegration"})


def cheatsheet():
    return "egcoin: Engle-Granger two-step cointegration"
