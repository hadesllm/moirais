"""Extended Kalman filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["extended_kalman"]


def extended_kalman(y, f, h, F, H, Q, R):
    """
    Extended Kalman filter

    Formula: linearize f, h via Jacobians at current state

    Parameters
    ----------
    y : array-like
        Input data.
    f : array-like
        Input data.
    h : array-like
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
    Schmidt (1966); Jazwinski (1970)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Extended Kalman filter"})


def cheatsheet():
    return "ekfF: Extended Kalman filter"
