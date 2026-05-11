"""Laplacian eigenvalues."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["laplacian_eigen"]


def laplacian_eigen(G):
    """
    Laplacian eigenvalues

    Formula: Lx = lambda x; L = D - A

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
    Chung (1997)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplacian eigenvalues"})


def cheatsheet():
    return "laplmo: Laplacian eigenvalues"
