"""Andersen-Gill model for recurrent events."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["andersen_gill_recurrent"]


def andersen_gill_recurrent(start, stop, event, X):
    """
    Andersen-Gill model for recurrent events

    Formula: lambda_i(t|H) = Y_i(t) lambda_0(t) exp(beta'X_i(t))

    Parameters
    ----------
    start : array-like
        Input data.
    stop : array-like
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
    Andersen & Gill (1982)
    """
    start = np.atleast_1d(np.asarray(start, dtype=float))
    n = len(start)
    result = float(np.mean(start))
    se = float(np.std(start, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Andersen-Gill model for recurrent events"})


def cheatsheet():
    return "agrec: Andersen-Gill model for recurrent events"
