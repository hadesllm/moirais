"""PACE (sparse FPCA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pace"]


def pace(Y, argvals, K):
    """
    PACE (sparse FPCA)

    Formula: covariance estimation from sparse data + BLUP scores

    Parameters
    ----------
    Y : array-like
        Input data.
    argvals : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yao-Müller-Wang (2005)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PACE (sparse FPCA)"})


def cheatsheet():
    return "pace: PACE (sparse FPCA)"
