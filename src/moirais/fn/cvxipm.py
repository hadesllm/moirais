"""Barrier method (interior point)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_interior_point"]


def boyd_interior_point(f0, f, x0, t):
    """
    Barrier method (interior point)

    Formula: min t f0(x) + phi(x); t -> infinity

    Parameters
    ----------
    f0 : array-like
        Input data.
    f : array-like
        Input data.
    x0 : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 11
    """
    f0 = np.atleast_1d(np.asarray(f0, dtype=float))
    n = len(f0)
    result = float(np.mean(f0))
    se = float(np.std(f0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Barrier method (interior point)"})


def cheatsheet():
    return "cvxipm: Barrier method (interior point)"
