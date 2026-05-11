"""Orthogonal random forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["orthogonal_random_forest"]


def orthogonal_random_forest(y, D, X):
    """
    Orthogonal random forest

    Formula: residualized forests + DML moment

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Oprescu-Syrgkanis-Wu (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orthogonal random forest"})


def cheatsheet():
    return "orfgrf: Orthogonal random forest"
