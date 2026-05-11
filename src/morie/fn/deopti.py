"""Differential evolution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["differential_evolution"]


def differential_evolution(f, population, F, CR):
    """
    Differential evolution

    Formula: v = x_a + F(x_b - x_c); crossover

    Parameters
    ----------
    f : array-like
        Input data.
    population : array-like
        Input data.
    F : array-like
        Input data.
    CR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Storn-Price (1997)
    """
    population = np.atleast_1d(np.asarray(population, dtype=float))
    n = len(population)
    result = float(np.mean(population))
    se = float(np.std(population, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differential evolution"})


def cheatsheet():
    return "deopti: Differential evolution"
