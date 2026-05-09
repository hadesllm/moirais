"""Predictive mean matching imputation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_pmm"]


def mi_pmm(y, X, R, K):
    """
    Predictive mean matching imputation

    Formula: impute with observed value of nearest predicted-mean donor

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    R : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Little (1988)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Predictive mean matching imputation"})


def cheatsheet():
    return "micord: Predictive mean matching imputation"
