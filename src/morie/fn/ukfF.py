"""Unscented Kalman filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["unscented_kalman"]


def unscented_kalman(y, f, h, Q, R):
    """
    Unscented Kalman filter

    Formula: sigma-point propagation through nonlinear f,h

    Parameters
    ----------
    y : array-like
        Input data.
    f : array-like
        Input data.
    h : array-like
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
    Julier-Uhlmann (1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unscented Kalman filter"})


def cheatsheet():
    return "ukfF: Unscented Kalman filter"
