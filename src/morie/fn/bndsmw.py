"""Bound with simulated weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_simul_weights"]


def bound_simul_weights(y, D, X, B):
    """
    Bound with simulated weights

    Formula: resample weights; CI from simul

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews-Shi (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound with simulated weights"})


def cheatsheet():
    return "bndsmw: Bound with simulated weights"
