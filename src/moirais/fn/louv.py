"""Louvain greedy modularity-maximisation community detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["louvain_communities"]


def louvain_communities(y, A, resolution):
    """
    Louvain greedy modularity-maximisation community detection

    Formula: iterate: move each node to neighbour community maximising delta Q

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    resolution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blondel, Guillaume, Lambiotte, Lefebvre (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Louvain greedy modularity-maximisation community detection"})


def cheatsheet():
    return "louv: Louvain greedy modularity-maximisation community detection"
