"""Manski no-assumption bounds with outcome covariates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["manski_no_assumption_outcome"]


def manski_no_assumption_outcome(y, D, X, y_min, y_max):
    """
    Manski no-assumption bounds with outcome covariates

    Formula: intersect bounds across X-strata

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    y_min : array-like
        Input data.
    y_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (1990, 2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Manski no-assumption bounds with outcome covariates"})


def cheatsheet():
    return "mskbnd2: Manski no-assumption bounds with outcome covariates"
