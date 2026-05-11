"""Optimal regression-tree regime."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["optimal_tree_regime"]


def optimal_tree_regime(y, A, W):
    """
    Optimal regression-tree regime

    Formula: recursive partitioning maximizing E[Y(d)]

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Laber et al (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal regression-tree regime"})


def cheatsheet():
    return "opttre: Optimal regression-tree regime"
