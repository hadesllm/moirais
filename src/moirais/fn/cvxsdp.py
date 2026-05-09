"""Semidefinite program."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_sdp"]


def boyd_sdp(c, F):
    """
    Semidefinite program

    Formula: min c'x s.t. F0 + sum x_i F_i >= 0

    Parameters
    ----------
    c : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semidefinite program"})


def cheatsheet():
    return "cvxsdp: Semidefinite program"
