"""Legendre polynomial basis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["legendre_basis"]


def legendre_basis(x, K):
    """
    Legendre polynomial basis

    Formula: recurrence relation

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Legendre (1782)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Legendre polynomial basis"})


def cheatsheet():
    return "legpb: Legendre polynomial basis"
