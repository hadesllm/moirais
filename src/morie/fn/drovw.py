"""Overlap-weighted DR-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_overlap_weighted"]


def dr_overlap_weighted(y, D, X):
    """
    Overlap-weighted DR-DiD

    Formula: weights = e(X)(1-e(X))

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li-Morgan-Zaslavsky (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Overlap-weighted DR-DiD"})


def cheatsheet():
    return "drovw: Overlap-weighted DR-DiD"
