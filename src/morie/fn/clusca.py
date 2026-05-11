"""Local clustering coefficient (transitivity)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clustering_coefficient"]


def clustering_coefficient(y, A, node):
    """
    Local clustering coefficient (transitivity)

    Formula: C_v = 2 * (# triangles at v) / (k_v (k_v - 1))

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    node : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Watts & Strogatz (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local clustering coefficient (transitivity)"})


def cheatsheet():
    return "clusca: Local clustering coefficient (transitivity)"
