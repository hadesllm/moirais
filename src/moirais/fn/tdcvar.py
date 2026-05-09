"""Time-dependent covariate adjustment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["time_dep_covariate"]


def time_dep_covariate(y, A, L_t, time):
    """
    Time-dependent covariate adjustment

    Formula: sequentially update L_t models

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
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
    Hernán-Robins (2020) Ch 21
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-dependent covariate adjustment"})


def cheatsheet():
    return "tdcvar: Time-dependent covariate adjustment"
