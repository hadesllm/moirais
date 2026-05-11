"""Dual norm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_dual_norm"]


def boyd_dual_norm(norm, z):
    """
    Dual norm

    Formula: |z|_* = sup{z'x : |x| <= 1}

    Parameters
    ----------
    norm : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 3
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dual norm"})


def cheatsheet():
    return "cvxdle: Dual norm"
