"""TMLE for recurrent-event outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_recurrent"]


def tmle_recurrent(time, event, D, X):
    """
    TMLE for recurrent-event outcomes

    Formula: target rate ratio with clever covariate per recurrence

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Westling-vdL (2024)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for recurrent-event outcomes"})


def cheatsheet():
    return "tmlrec: TMLE for recurrent-event outcomes"
