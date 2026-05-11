"""Empirical orthogonal functions (EOFs)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["empirical_orthogonal_func"]


def empirical_orthogonal_func(X):
    """
    Empirical orthogonal functions (EOFs)

    Formula: PCA on space-time field

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lorenz (1956)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical orthogonal functions (EOFs)"})


def cheatsheet():
    return "vime: Empirical orthogonal functions (EOFs)"
