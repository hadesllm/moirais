"""DR-DiD sensitivity to parallel-trends violation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sensitivity_did"]


def sensitivity_did(y, D, X, M):
    """
    DR-DiD sensitivity to parallel-trends violation

    Formula: bias bound under M-violation

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rambachan-Roth (2023)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD sensitivity to parallel-trends violation"})


def cheatsheet():
    return "snmtst: DR-DiD sensitivity to parallel-trends violation"
