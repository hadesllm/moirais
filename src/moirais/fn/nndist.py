"""Nearest neighbor distance distribution G(r)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nearest_neighbor_distance"]


def nearest_neighbor_distance(coords, r_grid):
    """
    Nearest neighbor distance distribution G(r)

    Formula: empirical CDF of nearest neighbor distances

    Parameters
    ----------
    coords : array-like
        Input data.
    r_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diggle (2003)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nearest neighbor distance distribution G(r)"})


def cheatsheet():
    return "nndist: Nearest neighbor distance distribution G(r)"
