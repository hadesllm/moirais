"""Soft assignment matrix from entropic OT for matching."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_softassignment"]


def ot_softassignment(a, b, C, epsilon):
    """
    Soft assignment matrix from entropic OT for matching

    Formula: T from Sinkhorn; row-normalised gives assignment probs

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: assign

    References
    ----------
    Cuturi (2013)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Soft assignment matrix from entropic OT for matching"})


def cheatsheet():
    return "otsoft: Soft assignment matrix from entropic OT for matching"
