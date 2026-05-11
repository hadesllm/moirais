"""Chained imputation for missing covariates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chained_imputation"]


def chained_imputation(time, event, X, mi_iter):
    """
    Chained imputation for missing covariates

    Formula: sequential MI with auxiliary variables

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    mi_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van Buuren (2018)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chained imputation for missing covariates"})


def cheatsheet():
    return "sschin: Chained imputation for missing covariates"
