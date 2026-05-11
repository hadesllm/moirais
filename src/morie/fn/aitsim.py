"""Gini-Simpson diversity index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_simpson"]


def compositional_simpson(x):
    """
    Gini-Simpson diversity index

    Formula: D(x) = 1 - Σ x_i²

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: D

    References
    ----------
    Simpson (1949)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gini-Simpson diversity index"})


def cheatsheet():
    return "aitsim: Gini-Simpson diversity index"
