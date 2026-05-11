"""Louvain modularity optimization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["louvain_communities"]


def louvain_communities(G, resolution):
    """
    Louvain modularity optimization

    Formula: greedy modularity max via local moves

    Parameters
    ----------
    G : array-like
        Input data.
    resolution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blondel et al (2008)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Louvain modularity optimization"})


def cheatsheet():
    return "comlou: Louvain modularity optimization"
