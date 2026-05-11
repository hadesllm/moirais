"""Higgins' I² heterogeneity index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_higgins_i2"]


def ma_higgins_i2(Q, k):
    """
    Higgins' I² heterogeneity index

    Formula: I² = max(0, (Q-(k-1))/Q)

    Parameters
    ----------
    Q : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: I2, ci

    References
    ----------
    Higgins & Thompson (2002)
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Higgins' I² heterogeneity index"})


def cheatsheet():
    return "mai2: Higgins' I² heterogeneity index"
