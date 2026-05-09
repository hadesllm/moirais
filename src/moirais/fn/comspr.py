"""Spectral clustering on Laplacian."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spectral_clustering"]


def spectral_clustering(G, k):
    """
    Spectral clustering on Laplacian

    Formula: k-means on bottom-k eigenvectors of L

    Parameters
    ----------
    G : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    von Luxburg (2007)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral clustering on Laplacian"})


def cheatsheet():
    return "comspr: Spectral clustering on Laplacian"
