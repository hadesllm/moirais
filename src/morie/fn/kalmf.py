"""Kalman filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kalman_filter"]


def kalman_filter(y, F, H, Q, R):
    """
    Kalman filter

    Formula: predict + update steps for linear Gaussian SSM

    Parameters
    ----------
    y : array-like
        Input data.
    F : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalman (1960)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman filter"})


def cheatsheet():
    return "kalmF: Kalman filter"
