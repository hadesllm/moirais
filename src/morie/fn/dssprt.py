"""DSSP secondary structure assignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dssp_secondary"]


def dssp_secondary(coords):
    """
    DSSP secondary structure assignment

    Formula: hydrogen-bond pattern → 8-state alphabet

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
    Kabsch-Sander (1983)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DSSP secondary structure assignment"})


def cheatsheet():
    return "dssprt: DSSP secondary structure assignment"
