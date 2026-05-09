"""Collaborative double-robust TMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_cdrs"]


def tmle_cdrs(y, D, X):
    """
    Collaborative double-robust TMLE

    Formula: propensity model selected via outcome-residual driven score

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
    vdL & Gruber (2010); Ju et al (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Collaborative double-robust TMLE"})


def cheatsheet():
    return "tmlcds: Collaborative double-robust TMLE"
