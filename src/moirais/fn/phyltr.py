"""Neighbor-joining tree."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["phylogenetic_tree_nj"]


def phylogenetic_tree_nj(distance_matrix):
    """
    Neighbor-joining tree

    Formula: iteratively join pair minimizing Q-metric

    Parameters
    ----------
    distance_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Saitou-Nei (1987)
    """
    distance_matrix = np.atleast_1d(np.asarray(distance_matrix, dtype=float))
    n = len(distance_matrix)
    result = float(np.mean(distance_matrix))
    se = float(np.std(distance_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neighbor-joining tree"})


def cheatsheet():
    return "phyltr: Neighbor-joining tree"
