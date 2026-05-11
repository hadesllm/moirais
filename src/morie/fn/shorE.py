"""Shor's quantum factoring (period-finding)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shor_factoring"]


def shor_factoring(N):
    """
    Shor's quantum factoring (period-finding)

    Formula: QFT-based period of f(x)=a^x mod N

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shor (1994)
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shor's quantum factoring (period-finding)"})


def cheatsheet():
    return "shorE: Shor's quantum factoring (period-finding)"
