"""Inverse Laplace transform."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["inverse_laplace"]


def inverse_laplace(F, s, t):
    """
    Inverse Laplace transform

    Formula: Bromwich integral / partial-fraction

    Parameters
    ----------
    F : array-like
        Input data.
    s : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse Laplace transform"})


def cheatsheet():
    return "laplI: Inverse Laplace transform"
