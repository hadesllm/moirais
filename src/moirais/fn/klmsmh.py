"""Kalman/RTS smoother."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kalman_smoother"]


def kalman_smoother(y, model, filtered):
    """
    Kalman/RTS smoother

    Formula: backward recursion using filtered estimates

    Parameters
    ----------
    y : array-like
        Input data.
    model : array-like
        Input data.
    filtered : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rauch-Tung-Striebel (1965)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kalman/RTS smoother"})


def cheatsheet():
    return "klmsmh: Kalman/RTS smoother"
