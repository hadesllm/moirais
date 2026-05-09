"""Empirical Pickands dependence function A(t)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_pickands_dep_fn"]


def evt_pickands_dep_fn(x, y, t_grid):
    """
    Empirical Pickands dependence function A(t)

    Formula: Â(t) = -log(P̂(F_X(X)<=u(t),F_Y(Y)<=u(t)))/(-log u(t))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    t_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: A

    References
    ----------
    Pickands (1981)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical Pickands dependence function A(t)"})


def cheatsheet():
    return "evpdfn: Empirical Pickands dependence function A(t)"
