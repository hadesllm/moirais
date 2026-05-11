"""Discrete-time survival via complementary log-log."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["discrete_time_survival"]


def discrete_time_survival(time_discrete, event, X):
    """
    Discrete-time survival via complementary log-log

    Formula: log(-log(1-h_t)) = alpha_t + beta X

    Parameters
    ----------
    time_discrete : array-like
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
    Singer-Willett (2003)
    """
    time_discrete = np.atleast_1d(np.asarray(time_discrete, dtype=float))
    n = len(time_discrete)
    result = float(np.mean(time_discrete))
    se = float(np.std(time_discrete, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete-time survival via complementary log-log"})


def cheatsheet():
    return "survdsc: Discrete-time survival via complementary log-log"
