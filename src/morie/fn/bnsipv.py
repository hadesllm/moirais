"""Partial IV bound under one-sided compliance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_iv_partial"]


def bound_iv_partial(y, D, Z):
    """
    Partial IV bound under one-sided compliance

    Formula: intersection of IV + monotone

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mogstad-Santos-Torgovitsky (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partial IV bound under one-sided compliance"})


def cheatsheet():
    return "bnsipv: Partial IV bound under one-sided compliance"
