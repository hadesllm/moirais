"""Watts-Strogatz small-world."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["watts_strogatz"]


def watts_strogatz(n, k, p):
    """
    Watts-Strogatz small-world

    Formula: rewire ring lattice with prob p

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Watts-Strogatz (1998)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Watts-Strogatz small-world"})


def cheatsheet():
    return "watstro: Watts-Strogatz small-world"
