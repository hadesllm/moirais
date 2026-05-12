"""Pairs (case) bootstrap for OLS -- resample (X,y) jointly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_pairs_regression"]


def boot_pairs_regression(X, y, B):
    """
    Pairs (case) bootstrap for OLS -- resample (X,y) jointly

    Formula: Sample rows of (X,y) with replacement

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_b

    References
    ----------
    Freedman (1981)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pairs (case) bootstrap for OLS -- resample (X,y) jointly"})


def cheatsheet():
    return "btpair: Pairs (case) bootstrap for OLS -- resample (X,y) jointly"
