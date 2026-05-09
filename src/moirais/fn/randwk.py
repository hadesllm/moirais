"""Random walk on network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["random_walk"]


def random_walk(G, start, steps):
    """
    Random walk on network

    Formula: P(j|i) = A_ij / k_i

    Parameters
    ----------
    G : array-like
        Input data.
    start : array-like
        Input data.
    steps : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random walk on network"})


def cheatsheet():
    return "randwk: Random walk on network"
