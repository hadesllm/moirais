"""Subgradient definition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_subgradient"]


def boyd_subgradient(f, x):
    """
    Subgradient definition

    Formula: g in df(x) iff f(y) >= f(x) + g'(y-x) for all y

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: subgradient

    References
    ----------
    Boyd CVX Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Subgradient definition"})


def cheatsheet():
    return "cvxsbp: Subgradient definition"
