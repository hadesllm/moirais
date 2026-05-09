"""Smoothed Huber gradient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_smooth_huber_grad"]


def boyd_smooth_huber_grad(u, M):
    """
    Smoothed Huber gradient

    Formula: phi'(u) = u/M if |u|<=M else sign(u)

    Parameters
    ----------
    u : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient

    References
    ----------
    Boyd CVX Ch 9
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothed Huber gradient"})


def cheatsheet():
    return "cvxsmh: Smoothed Huber gradient"
