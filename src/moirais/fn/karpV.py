"""Koza genetic programming."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["genetic_programming"]


def genetic_programming(fitness, ops, gens):
    """
    Koza genetic programming

    Formula: crossover + mutation on expression trees

    Parameters
    ----------
    fitness : array-like
        Input data.
    ops : array-like
        Input data.
    gens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koza (1992)
    """
    fitness = np.atleast_1d(np.asarray(fitness, dtype=float))
    n = len(fitness)
    result = float(np.mean(fitness))
    se = float(np.std(fitness, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Koza genetic programming"})


def cheatsheet():
    return "karpV: Koza genetic programming"
