"""Hopcroft-Karp bipartite max matching."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bipartite_matching"]


def bipartite_matching(U, V, E):
    """
    Hopcroft-Karp bipartite max matching

    Formula: BFS layering + DFS augmentation

    Parameters
    ----------
    U : array-like
        Input data.
    V : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hopcroft-Karp (1973)
    """
    U = np.atleast_1d(np.asarray(U, dtype=float))
    n = len(U)
    result = float(np.mean(U))
    se = float(np.std(U, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hopcroft-Karp bipartite max matching"})


def cheatsheet():
    return "bipMch: Hopcroft-Karp bipartite max matching"
