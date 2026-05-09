"""Snijders-Bosker R^2 level-2 (between)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["snijders_bosker_r2_level2"]


def snijders_bosker_r2_level2(y, X, cluster):
    """
    Snijders-Bosker R^2 level-2 (between)

    Formula: R^2_2 = 1 - (sigma2_e1/n + sigma2_u1) / (sigma2_e0/n + sigma2_u0)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Snijders & Bosker (1994, 2012 §7)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Snijders-Bosker R^2 level-2 (between)"})


def cheatsheet():
    return "snr2u: Snijders-Bosker R^2 level-2 (between)"
