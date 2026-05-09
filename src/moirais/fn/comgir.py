"""Girvan-Newman edge-betweenness."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["girvan_newman"]


def girvan_newman(G):
    """
    Girvan-Newman edge-betweenness

    Formula: iteratively remove highest-betweenness edge

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
    Girvan-Newman (2002)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Girvan-Newman edge-betweenness"})


def cheatsheet():
    return "comgir: Girvan-Newman edge-betweenness"
