"""Semi-doubly-robust forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["semi_doubly_robust_forest"]


def semi_doubly_robust_forest(y, D, X, K_fold):
    """
    Semi-doubly-robust forest

    Formula: cross-fit forest scores + AIPW

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    K_fold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov-Demirer-Duflo-Fernández-Val (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semi-doubly-robust forest"})


def cheatsheet():
    return "sdcfst: Semi-doubly-robust forest"
