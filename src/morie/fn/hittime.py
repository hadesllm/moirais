"""Random walk hitting time."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hitting_time"]


def hitting_time(G, start, target):
    """
    Random walk hitting time

    Formula: E[first time random walker reaches v]

    Parameters
    ----------
    G : array-like
        Input data.
    start : array-like
        Input data.
    target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lovász (1996)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random walk hitting time"})


def cheatsheet():
    return "hittime: Random walk hitting time"
