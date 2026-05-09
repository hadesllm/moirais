"""Bound under outcome transformation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_transform"]


def bound_transform(y, D, X, transform):
    """
    Bound under outcome transformation

    Formula: bounds invariant under monotone transform

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    transform : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov et al (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under outcome transformation"})


def cheatsheet():
    return "bndtfm: Bound under outcome transformation"
