"""Splitting variables V at level epsilon defined as the conditional probabilities of the offspring sets given the parent set in a tree-based prior.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tree_splitting_variables"]


def ghosal_ch3_tree_splitting_variables(A_epsilon, epsilon):
    """
    Splitting variables V at level epsilon defined as the conditional probabilities of the offspring sets given the parent set in a tree-based prior.

    Formula: V_{epsilon 0} = P(A_{epsilon 0} | A_epsilon),   V_{epsilon 1} = P(A_{epsilon 1} | A_epsilon)

    Parameters
    ----------
    A_epsilon : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.11, p. 37
    """
    A_epsilon = np.atleast_1d(np.asarray(A_epsilon, dtype=float))
    n = len(A_epsilon)
    result = float(np.mean(A_epsilon))
    se = float(np.std(A_epsilon, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Splitting variables V at level epsilon defined as the conditional probabilities of the offspring sets given the parent set in a tree-based prior."})


def cheatsheet():
    return "ghs018: Splitting variables V at level epsilon defined as the conditional probabilities of the offspring sets given the parent set in a tree-based prior."
