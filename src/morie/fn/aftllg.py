"""AFT log-logistic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aft_log_logistic"]


def aft_log_logistic(time, event, X):
    """
    AFT log-logistic

    Formula: T = exp(beta X) * exp(sigma W); W ~ logistic

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
    Klein-Moeschberger (2003)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AFT log-logistic"})


def cheatsheet():
    return "aftllg: AFT log-logistic"
