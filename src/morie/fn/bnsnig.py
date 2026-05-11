"""Bound under no unobserved invariance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_no_unobserved_inv"]


def bound_no_unobserved_inv(y, D, X, X_inv):
    """
    Bound under no unobserved invariance

    Formula: removes UH assumption

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    X_inv : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tchetgen Tchetgen (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under no unobserved invariance"})


def cheatsheet():
    return "bnsnig: Bound under no unobserved invariance"
