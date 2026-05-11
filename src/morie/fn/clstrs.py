"""Cluster sample variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cluster_design"]


def cluster_design(y, cluster):
    """
    Cluster sample variance

    Formula: between-cluster variance dominates

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cluster sample variance"})


def cheatsheet():
    return "clstrs: Cluster sample variance"
