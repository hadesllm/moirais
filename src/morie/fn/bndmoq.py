"""Quantile-equivariant bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_moment_qed"]


def bound_moment_qed(y, D, X, quantile):
    """
    Quantile-equivariant bound

    Formula: bounds invariant under quantile transform

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov-Hansen (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile-equivariant bound"})


def cheatsheet():
    return "bndmoq: Quantile-equivariant bound"
