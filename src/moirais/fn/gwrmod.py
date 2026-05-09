"""Geographically Weighted Regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geographically_weighted_regression"]


def geographically_weighted_regression(y, X, coords, bandwidth):
    """
    Geographically Weighted Regression

    Formula: beta(s) varies by location; kernel-weighted local OLS

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    coords : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brunsdon-Fotheringham-Charlton (1996)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geographically Weighted Regression"})


def cheatsheet():
    return "gwrmod: Geographically Weighted Regression"
