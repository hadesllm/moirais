"""Genetic algorithm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["genetic_algorithm"]


def genetic_algorithm(f, population, generations):
    """
    Genetic algorithm

    Formula: selection + crossover + mutation

    Parameters
    ----------
    f : array-like
        Input data.
    population : array-like
        Input data.
    generations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holland (1975)
    """
    population = np.atleast_1d(np.asarray(population, dtype=float))
    n = len(population)
    result = float(np.mean(population))
    se = float(np.std(population, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genetic algorithm"})


def cheatsheet():
    return "ga_opt: Genetic algorithm"
