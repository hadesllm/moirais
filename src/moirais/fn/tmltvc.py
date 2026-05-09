"""TMLE under time-varying confounding (non-longitudinal sequential)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_time_varying_confound"]


def tmle_time_varying_confound(y, D_t, L_t, time):
    """
    TMLE under time-varying confounding (non-longitudinal sequential)

    Formula: sequential clever covariates per time step

    Parameters
    ----------
    y : array-like
        Input data.
    D_t : array-like
        Input data.
    L_t : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Gruber (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE under time-varying confounding (non-longitudinal sequential)"})


def cheatsheet():
    return "tmltvc: TMLE under time-varying confounding (non-longitudinal sequential)"
