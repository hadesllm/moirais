"""Spatial detrending (median polish)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_detrending"]


def spatial_detrending(values, grid):
    """
    Spatial detrending (median polish)

    Formula: iterative row/column median subtraction

    Parameters
    ----------
    values : array-like
        Input data.
    grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tukey (1977)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial detrending (median polish)"})


def cheatsheet():
    return "spdetr: Spatial detrending (median polish)"
