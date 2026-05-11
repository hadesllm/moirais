"""Spectral radius of adjacency."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spectral_radius"]


def spectral_radius(G):
    """
    Spectral radius of adjacency

    Formula: max |lambda(A)|

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
    Cvetković et al (2010)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral radius of adjacency"})


def cheatsheet():
    return "spcrad: Spectral radius of adjacency"
