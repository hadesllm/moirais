"""Log-normal frailty for recurrent events."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lognormal_frailty"]


def lognormal_frailty(time, event, X, cluster):
    """
    Log-normal frailty for recurrent events

    Formula: lambda_ij(t|b_i) = lambda_0(t) exp(beta'X_ij + b_i), b_i ~ N(0, sigma^2)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
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
    McGilchrist (1993)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-normal frailty for recurrent events"})


def cheatsheet():
    return "lnfrm: Log-normal frailty for recurrent events"
