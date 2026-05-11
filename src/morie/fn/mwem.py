"""Multiplicative weights exponential mechanism."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mwem"]


def mwem(queries, epsilon, T):
    """
    Multiplicative weights exponential mechanism

    Formula: alternate ExpM query selection + MW update

    Parameters
    ----------
    queries : array-like
        Input data.
    epsilon : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hardt-Ligett-McSherry (2012)
    """
    queries = np.atleast_1d(np.asarray(queries, dtype=float))
    n = len(queries)
    result = float(np.mean(queries))
    se = float(np.std(queries, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiplicative weights exponential mechanism"})


def cheatsheet():
    return "mwem: Multiplicative weights exponential mechanism"
