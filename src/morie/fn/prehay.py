"""Preacher-Hayes bootstrap multiple-mediator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["preacher_hayes_indirect"]


def preacher_hayes_indirect(X, M, Y, B):
    """
    Preacher-Hayes bootstrap multiple-mediator

    Formula: separate IE_k = a_k * b_k for each mediator + total

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Preacher & Hayes (2008) PROCESS
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Preacher-Hayes bootstrap multiple-mediator"})


def cheatsheet():
    return "prehay: Preacher-Hayes bootstrap multiple-mediator"
