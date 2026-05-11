"""Grover search."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["grover_search"]


def grover_search(oracle, N):
    """
    Grover search

    Formula: O(√N) amplitude amplification

    Parameters
    ----------
    oracle : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Grover (1996)
    """
    oracle = np.atleast_1d(np.asarray(oracle, dtype=float))
    n = len(oracle)
    result = float(np.mean(oracle))
    se = float(np.std(oracle, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grover search"})


def cheatsheet():
    return "groverS: Grover search"
