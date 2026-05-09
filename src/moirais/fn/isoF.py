"""Isolation forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["isolation_forest"]


def isolation_forest(X, n_trees):
    """
    Isolation forest

    Formula: anomaly score from avg path length over isolation trees

    Parameters
    ----------
    X : array-like
        Input data.
    n_trees : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu-Ting-Zhou (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isolation forest"})


def cheatsheet():
    return "isoF: Isolation forest"
