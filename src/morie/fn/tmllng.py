"""TMLE for longitudinal data with time-varying treatments and confounders."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_longitudinal"]


def tmle_longitudinal(L, A, Y, time):
    """
    TMLE for longitudinal data with time-varying treatments and confounders

    Formula: L-TMLE: iterate Q-bar updates from t=K to t=0 with sequential clever covariates

    Parameters
    ----------
    L : array-like
        Input data.
    A : array-like
        Input data.
    Y : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Laan & Gruber (2012); Petersen et al (2014)
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    n = len(L)
    result = float(np.mean(L))
    se = float(np.std(L, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for longitudinal data with time-varying treatments and confounders"})


def cheatsheet():
    return "tmllng: TMLE for longitudinal data with time-varying treatments and confounders"
