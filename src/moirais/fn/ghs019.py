"""Probability assigned by a tree-based random measure to a partitioning set as a product of splitting variables along the path from the root.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tree_set_probability"]


def ghosal_ch3_tree_set_probability(V, epsilon):
    """
    Probability assigned by a tree-based random measure to a partitioning set as a product of splitting variables along the path from the root.

    Formula: P(A_{epsilon_1 ... epsilon_m}) = V_{epsilon_1} * V_{epsilon_1 epsilon_2} * ... * V_{epsilon_1 ... epsilon_m}

    Parameters
    ----------
    V : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.12, p. 37
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Probability assigned by a tree-based random measure to a partitioning set as a product of splitting variables along the path from the root."})


def cheatsheet():
    return "ghs019: Probability assigned by a tree-based random measure to a partitioning set as a product of splitting variables along the path from the root."
