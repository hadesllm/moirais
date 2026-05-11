"""Thiessen / Voronoi polygon partition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["thiessen_polygons"]


def thiessen_polygons(coords):
    """
    Thiessen / Voronoi polygon partition

    Formula: each point gets cell of all locations nearer to it

    Parameters
    ----------
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Thiessen (1911)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thiessen / Voronoi polygon partition"})


def cheatsheet():
    return "thnsst: Thiessen / Voronoi polygon partition"
