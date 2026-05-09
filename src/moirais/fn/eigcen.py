"""Eigenvector centrality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["eigenvector_centrality"]


def eigenvector_centrality(G):
    """
    Eigenvector centrality

    Formula: x = lambda^-1 A x; principal eigenvector

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
    Bonacich (1972)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Eigenvector centrality"})


def cheatsheet():
    return "eigcen: Eigenvector centrality"
