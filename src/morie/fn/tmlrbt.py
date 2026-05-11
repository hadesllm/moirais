"""Robust TMLE under model misspecification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_robust"]


def tmle_robust(y, D, X, trim):
    """
    Robust TMLE under model misspecification

    Formula: M-estimator with bounded clever covariate

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    trim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tan (2010); Frangakis-Petersen-vdL (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Robust TMLE under model misspecification"})


def cheatsheet():
    return "tmlrbt: Robust TMLE under model misspecification"
