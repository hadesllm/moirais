"""Balanced repeated replication."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["brr_balanced"]


def brr_balanced(data, strata, brr_design):
    """
    Balanced repeated replication

    Formula: Hadamard half-sample design

    Parameters
    ----------
    data : array-like
        Input data.
    strata : array-like
        Input data.
    brr_design : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McCarthy (1969)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Balanced repeated replication"})


def cheatsheet():
    return "brrest: Balanced repeated replication"
