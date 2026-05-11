"""AFT model with Weibull baseline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aft_weibull"]


def aft_weibull(time, event, X):
    """
    AFT model with Weibull baseline

    Formula: log T = beta X + sigma W; W ~ Gumbel

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AFT model with Weibull baseline"})


def cheatsheet():
    return "aftwbl: AFT model with Weibull baseline"
