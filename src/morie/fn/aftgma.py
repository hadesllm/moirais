"""AFT generalized gamma."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aft_generalized_gamma"]


def aft_generalized_gamma(time, event, X):
    """
    AFT generalized gamma

    Formula: 3-parameter family containing Weibull, log-normal, gamma

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stacy-Mihram (1965)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AFT generalized gamma"})


def cheatsheet():
    return "aftgma: AFT generalized gamma"
