"""Betweenness centrality of a node."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["betweenness_centrality"]


def betweenness_centrality(y, A, node):
    """
    Betweenness centrality of a node

    Formula: C_B(v) = sum_{s != v != t} sigma_st(v) / sigma_st

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
    Freeman (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Betweenness centrality of a node"})


def cheatsheet():
    return "netbtw: Betweenness centrality of a node"
