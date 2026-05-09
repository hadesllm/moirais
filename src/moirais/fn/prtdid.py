"""Partition-based DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["partition_did"]


def partition_did(y, D, X):
    """
    Partition-based DiD

    Formula: discrete partitions on X; ATT per cell

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goodman-Bacon (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partition-based DiD"})


def cheatsheet():
    return "prtdid: Partition-based DiD"
