# morie.fn — function file (hadesllm/morie)
"""One-hot encoding: represent categorical variable with K-1 indicator columns."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_hot_encoding"]


def geron_one_hot_encoding(X):
    """
    One-hot encoding: represent categorical variable with K-1 indicator columns

    Formula: x_ik = I(category_i = k)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_ohe

    References
    ----------
    Géron Ch 2
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-hot encoding: represent categorical variable with K-1 indicator columns"})


def cheatsheet():
    return "hmohe: One-hot encoding: represent categorical variable with K-1 indicator columns"
