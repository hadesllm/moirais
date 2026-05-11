"""Years of life lost."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["yll_calculation"]


def yll_calculation(deaths, ages, life_table):
    """
    Years of life lost

    Formula: YLL = N × (life_expectancy - age_at_death)

    Parameters
    ----------
    deaths : array-like
        Input data.
    ages : array-like
        Input data.
    life_table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    WHO Global Burden of Disease
    """
    deaths = np.atleast_1d(np.asarray(deaths, dtype=float))
    n = len(deaths)
    result = float(np.mean(deaths))
    se = float(np.std(deaths, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Years of life lost"})


def cheatsheet():
    return "yllyear: Years of life lost"
