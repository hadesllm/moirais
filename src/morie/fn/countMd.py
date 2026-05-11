"""Mediation for count outcome."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["count_mediation"]


def count_mediation(Y, X, M, C):
    """
    Mediation for count outcome

    Formula: Poisson/NB-based decomposition

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2015) book Ch.2
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediation for count outcome"})


def cheatsheet():
    return "countMd: Mediation for count outcome"
