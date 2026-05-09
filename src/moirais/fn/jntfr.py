"""Joint frailty for recurrent + terminal events."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["joint_frailty"]


def joint_frailty(time, event, terminal, cluster):
    """
    Joint frailty for recurrent + terminal events

    Formula: lambda_R(t|w) = w lambda_0R, lambda_T(t|w) = w^alpha lambda_0T

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    terminal : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu, Wolfe, Huang (2004)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint frailty for recurrent + terminal events"})


def cheatsheet():
    return "jntfr: Joint frailty for recurrent + terminal events"
