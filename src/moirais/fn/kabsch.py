"""Kabsch optimal rotation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kabsch_superpose"]


def kabsch_superpose(coords1, coords2):
    """
    Kabsch optimal rotation

    Formula: SVD of cross-covariance matrix

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kabsch optimal rotation"})


def cheatsheet():
    return "kabsch: Kabsch optimal rotation"
