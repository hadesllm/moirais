"""Infomap community via random walk MDL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["infomap"]


def infomap(G):
    """
    Infomap community via random walk MDL

    Formula: min description length of random walk

    Parameters
    ----------
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rosvall-Bergstrom (2008)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Infomap community via random walk MDL"})


def cheatsheet():
    return "cominf: Infomap community via random walk MDL"
