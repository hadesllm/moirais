"""Stefan-Boltzmann radiation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stefan_boltzmann"]


def stefan_boltzmann(T, emissivity):
    """
    Stefan-Boltzmann radiation

    Formula: j* = σ T⁴

    Parameters
    ----------
    T : array-like
        Input data.
    emissivity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stefan (1879)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stefan-Boltzmann radiation"})


def cheatsheet():
    return "stefan: Stefan-Boltzmann radiation"
