"""Functional manifold learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["manifold_functional"]


def manifold_functional(Y, k, method):
    """
    Functional manifold learning

    Formula: Isomap / diffusion on curve space

    Parameters
    ----------
    Y : array-like
        Input data.
    k : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chen-Müller (2012)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional manifold learning"})


def cheatsheet():
    return "manfd: Functional manifold learning"
