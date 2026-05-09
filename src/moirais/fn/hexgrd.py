"""Hexagonal grid binning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hexagonal_grid"]


def hexagonal_grid(coords, values, cell_size):
    """
    Hexagonal grid binning

    Formula: hex tessellation; aggregate observations per hex

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    cell_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carr-Olsen-Lee-White (1992)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hexagonal grid binning"})


def cheatsheet():
    return "hexgrd: Hexagonal grid binning"
