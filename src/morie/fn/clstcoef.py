"""Local clustering coefficient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clustering_coefficient"]


def clustering_coefficient(G):
    """
    Local clustering coefficient

    Formula: C_v = 2 e(N_v) / (k_v (k_v-1))

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
    Watts-Strogatz (1998)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local clustering coefficient"})


def cheatsheet():
    return "clstcoef: Local clustering coefficient"
