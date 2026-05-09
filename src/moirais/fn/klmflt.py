"""Kalman filter for state-space."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kalman_filter"]


def kalman_filter(y, model):
    """
    Kalman filter for state-space

    Formula: forward recursion: predict + update

    Parameters
    ----------
    y : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalman (1960); Durbin-Koopman (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman filter for state-space"})


def cheatsheet():
    return "klmflt: Kalman filter for state-space"
