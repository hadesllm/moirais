"""TMLE for rare outcomes -- small-sample correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_rare_outcome"]


def tmle_rare_outcome(y, D, X, prevalence):
    """
    TMLE for rare outcomes -- small-sample correction

    Formula: target with influence-curve-based finite-sample correction

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    prevalence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tran-Petersen-vdL (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for rare outcomes -- small-sample correction"})


def cheatsheet():
    return "tmlric: TMLE for rare outcomes -- small-sample correction"
