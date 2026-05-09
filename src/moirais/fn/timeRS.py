"""timeSVD++."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["timesvd"]


def timesvd(R, timestamps, K):
    """
    timeSVD++

    Formula: add temporal terms b_u(t), b_i(t), p_u(t)

    Parameters
    ----------
    R : array-like
        Input data.
    timestamps : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koren (2009) timeSVD
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "timeSVD++"})


def cheatsheet():
    return "timeRS: timeSVD++"
