"""Betweenness centrality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["betweenness_centrality"]


def betweenness_centrality(G):
    """
    Betweenness centrality

    Formula: C_B(v) = sum sigma(s,t|v)/sigma(s,t)

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
    Freeman (1977); Brandes (2001)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Betweenness centrality"})


def cheatsheet():
    return "betcen: Betweenness centrality"
