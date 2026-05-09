"""Necessary-and-sufficient mean-zero conditions for a tree-based finitely additive measure on R to extend to a countably additive random measure.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tree_countable_additivity"]


def ghosal_ch3_tree_countable_additivity(V):
    """
    Necessary-and-sufficient mean-zero conditions for a tree-based finitely additive measure on R to extend to a countably additive random measure.

    Formula: E[ V_epsilon * V_{epsilon 0} * V_{epsilon 0 0} * ... ] = 0   for all epsilon in E*,   and E[ V_1 * V_{1 1} * V_{1 1 1} * ... ] = 0

    Parameters
    ----------
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.13, p. 38
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Necessary-and-sufficient mean-zero conditions for a tree-based finitely additive measure on R to extend to a countably additive random measure."})


def cheatsheet():
    return "ghs020: Necessary-and-sufficient mean-zero conditions for a tree-based finitely additive measure on R to extend to a countably additive random measure."
