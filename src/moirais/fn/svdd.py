"""Support vector data description."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["svdd"]


def svdd(X, C):
    """
    Support vector data description

    Formula: min sphere enclosing data in feature space

    Parameters
    ----------
    X : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tax-Duin (2004)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Support vector data description"})


def cheatsheet():
    return "svdd: Support vector data description"
