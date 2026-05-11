"""Outcome-only TMLE — robust if g misspecified."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_outcome_only_regr"]


def tmle_outcome_only_regr(y, D, X):
    """
    Outcome-only TMLE — robust if g misspecified

    Formula: propensity-free target via outcome regression alone

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
    Tan (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Outcome-only TMLE — robust if g misspecified"})


def cheatsheet():
    return "tmlovo: Outcome-only TMLE — robust if g misspecified"
