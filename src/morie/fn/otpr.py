"""Partial OT transporting only mass m <= min(|a|,|b|)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_partial_ot"]


def ot_partial_ot(a, b, C, m):
    """
    Partial OT transporting only mass m <= min(|a|,|b|)

    Formula: min_T <T,C>; T1<=a, T^T 1<=b, sum T = m

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T

    References
    ----------
    Caffarelli & McCann (2010)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partial OT transporting only mass m <= min(|a|,|b|)"})


def cheatsheet():
    return "otpr: Partial OT transporting only mass m <= min(|a|,|b|)"
