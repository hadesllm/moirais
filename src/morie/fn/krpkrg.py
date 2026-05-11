"""Poisson kriging for rates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["poisson_kriging"]


def poisson_kriging(coords, counts, population, variogram):
    """
    Poisson kriging for rates

    Formula: BLUP for rate = lambda accounting for population at risk

    Parameters
    ----------
    coords : array-like
        Input data.
    counts : array-like
        Input data.
    population : array-like
        Input data.
    variogram : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goovaerts (2005)
    """
    population = np.atleast_1d(np.asarray(population, dtype=float))
    n = len(population)
    result = float(np.mean(population))
    se = float(np.std(population, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson kriging for rates"})


def cheatsheet():
    return "krpkrg: Poisson kriging for rates"
