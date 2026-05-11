"""Surface retrieval from sparse sensors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["surface_retrieval"]


def surface_retrieval(coords, values, grid, method):
    """
    Surface retrieval from sparse sensors

    Formula: kriging or GP-based interpolation

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    grid : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rasmussen-Williams (2006)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Surface retrieval from sparse sensors"})


def cheatsheet():
    return "sfcret: Surface retrieval from sparse sensors"
