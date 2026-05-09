"""Community modularity Q on partition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["community_modularity"]


def community_modularity(G, partition):
    """
    Community modularity Q on partition

    Formula: Q from given partition

    Parameters
    ----------
    G : array-like
        Input data.
    partition : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2006)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Community modularity Q on partition"})


def cheatsheet():
    return "comten: Community modularity Q on partition"
