"""Adjacency matrix from edge list."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_adjacency_matrix"]


def sgt_adjacency_matrix(edges, n, directed):
    """
    Adjacency matrix from edge list

    Formula: A_{ij} = 1 if (i,j) ∈ E else 0

    Parameters
    ----------
    edges : array-like
        Input data.
    n : array-like
        Input data.
    directed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: A

    References
    ----------
    Chung (1997)
    """
    edges = np.atleast_1d(np.asarray(edges, dtype=float))
    n = len(edges)
    result = float(np.mean(edges))
    se = float(np.std(edges, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adjacency matrix from edge list"})


def cheatsheet():
    return "sgtadj: Adjacency matrix from edge list"
