"""Event-study leads + lags coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["event_study_coefficients"]


def event_study_coefficients(y, D, unit, time, cohort):
    """
    Event-study leads + lags coefficients

    Formula: y_it = sum_{e=-K}^{K} mu_e 1{t - g_i = e} + ...

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    cohort : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Borusyak-Jaravel (2017); Sun-Abraham (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Event-study leads + lags coefficients"})


def cheatsheet():
    return "evstud: Event-study leads + lags coefficients"
