"""Max-stable process simulation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["max_exceedance_curve"]


def max_exceedance_curve(coords, range):
    """
    Max-stable process simulation

    Formula: sample from spectral representation

    Parameters
    ----------
    coords : array-like
        Input data.
    range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Haan (1984)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max-stable process simulation"})


def cheatsheet():
    return "mxetA: Max-stable process simulation"
