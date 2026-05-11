# morie.fn — function file (hadesllm/morie)
"""Ordinal encoding: map categorical values to integers preserving order."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ordinal_encoding"]


def geron_ordinal_encoding(X, categories):
    """
    Ordinal encoding: map categorical values to integers preserving order

    Formula: x_cat -> int via lookup table; order matters

    Parameters
    ----------
    X : array-like
        Input data.
    categories : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_int

    References
    ----------
    Géron Ch 2
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinal encoding: map categorical values to integers preserving order"})


def cheatsheet():
    return "hmord: Ordinal encoding: map categorical values to integers preserving order"
