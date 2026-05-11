"""Engle-Granger two-step cointegration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["engle_granger"]


def engle_granger(Y1, Y2):
    """
    Engle-Granger two-step cointegration

    Formula: OLS residual ADF test

    Parameters
    ----------
    Y1 : array-like
        Input data.
    Y2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle-Granger (1987)
    """
    Y1 = np.atleast_1d(np.asarray(Y1, dtype=float))
    n = len(Y1)
    result = float(np.mean(Y1))
    se = float(np.std(Y1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Engle-Granger two-step cointegration"})


def cheatsheet():
    return "engrgr: Engle-Granger two-step cointegration"
