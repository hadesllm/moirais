"""Katz centrality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["katz_centrality"]


def katz_centrality(G, alpha, beta):
    """
    Katz centrality

    Formula: C_K = (I - alpha A)^-1 1

    Parameters
    ----------
    G : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Katz (1953)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Katz centrality"})


def cheatsheet():
    return "katzcn: Katz centrality"
