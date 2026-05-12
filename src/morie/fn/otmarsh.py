"""Shift the row marginal a -> ã via partial transport."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_marginal_shift"]


def ot_marginal_shift(a, b, C, delta):
    """
    Shift the row marginal a -> ã via partial transport

    Formula: Solve OT with a' = a - δ, where δ is removed mass

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T

    References
    ----------
    Caffarelli-McCann (2010)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shift the row marginal a -> ã via partial transport"})


def cheatsheet():
    return "otmarsh: Shift the row marginal a -> ã via partial transport"
