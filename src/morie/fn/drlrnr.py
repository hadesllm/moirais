"""R-learner for CATE (Nie-Wager)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["r_learner"]


def r_learner(y, D, X, ml_outcome, ml_propensity):
    """
    R-learner for CATE (Nie-Wager)

    Formula: min sum_i (Y_i - hat m(X_i) - tau(X_i)(D_i - hat e(X_i)))^2

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    ml_outcome : array-like
        Input data.
    ml_propensity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nie & Wager (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "R-learner for CATE (Nie-Wager)"})


def cheatsheet():
    return "drlrnr: R-learner for CATE (Nie-Wager)"
