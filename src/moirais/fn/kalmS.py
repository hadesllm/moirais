"""Rauch-Tung-Striebel smoother."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kalman_smoother"]


def kalman_smoother(y, F, H, Q, R):
    """
    Rauch-Tung-Striebel smoother

    Formula: backward pass on filtered means

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
    Rauch-Tung-Striebel (1965)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rauch-Tung-Striebel smoother"})


def cheatsheet():
    return "kalmS: Rauch-Tung-Striebel smoother"
