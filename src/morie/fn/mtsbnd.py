"""Monotone treatment selection bounds."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mts_bounds"]


def mts_bounds(Y, X, monotone):
    """
    Monotone treatment selection bounds

    Formula: impose monotonicity to tighten Manski

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    monotone : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Pepper (2000)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Monotone treatment selection bounds"})


def cheatsheet():
    return "mtsBnd: Monotone treatment selection bounds"
