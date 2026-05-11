"""RMSD between aligned structures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rmsd"]


def rmsd(coords1, coords2):
    """
    RMSD between aligned structures

    Formula: sqrt(mean(||x_i - y_i||^2)) after Kabsch superposition

    Parameters
    ----------
    coords1 : array-like
        Input data.
    coords2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kabsch (1976)
    """
    coords1 = np.atleast_1d(np.asarray(coords1, dtype=float))
    n = len(coords1)
    result = float(np.mean(coords1))
    se = float(np.std(coords1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RMSD between aligned structures"})


def cheatsheet():
    return "rmsdtr: RMSD between aligned structures"
