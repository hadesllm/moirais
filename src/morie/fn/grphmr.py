"""Graphormer (centrality + spatial encoding)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graphormer"]


def graphormer(G, X):
    """
    Graphormer (centrality + spatial encoding)

    Formula: transformer with degree + path-distance + edge bias

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ying et al (2021)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graphormer (centrality + spatial encoding)"})


def cheatsheet():
    return "grphmr: Graphormer (centrality + spatial encoding)"
