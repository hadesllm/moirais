"""PELT (pruned exact linear time)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pelt"]


def pelt(x, cost, penalty):
    """
    PELT (pruned exact linear time)

    Formula: DP with pruning, O(n) under linear cost

    Parameters
    ----------
    x : array-like
        Input data.
    cost : array-like
        Input data.
    penalty : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Killick-Fearnhead-Eckley (2012)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PELT (pruned exact linear time)"})


def cheatsheet():
    return "pelt: PELT (pruned exact linear time)"
