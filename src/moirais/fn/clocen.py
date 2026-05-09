"""Closeness centrality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["closeness_centrality"]


def closeness_centrality(G):
    """
    Closeness centrality

    Formula: C_C(v) = (n-1) / sum d(v,u)

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
    Bavelas (1950); Freeman (1979)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closeness centrality"})


def cheatsheet():
    return "clocen: Closeness centrality"
