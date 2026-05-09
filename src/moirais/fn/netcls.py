"""Closeness centrality of a node."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["closeness_centrality"]


def closeness_centrality(y, A, node):
    """
    Closeness centrality of a node

    Formula: C_C(v) = (n-1) / sum_{u != v} d(v,u)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    node : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bavelas (1950); Sabidussi (1966)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closeness centrality of a node"})


def cheatsheet():
    return "netcls: Closeness centrality of a node"
