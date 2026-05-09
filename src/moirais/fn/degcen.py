"""Degree centrality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["degree_centrality"]


def degree_centrality(G):
    """
    Degree centrality

    Formula: C_D(v) = deg(v) / (n-1)

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
    Freeman (1979)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Degree centrality"})


def cheatsheet():
    return "degcen: Degree centrality"
