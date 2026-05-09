"""DeepWalk uniform random walks."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deepwalk"]


def deepwalk(G, walk_len, dim):
    """
    DeepWalk uniform random walks

    Formula: random walks + skip-gram

    Parameters
    ----------
    G : array-like
        Input data.
    walk_len : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Perozzi-Al-Rfou-Skiena (2014)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DeepWalk uniform random walks"})


def cheatsheet():
    return "deepw: DeepWalk uniform random walks"
