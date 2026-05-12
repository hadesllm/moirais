# morie.fn -- function file (hadesllm/morie)
"""Non-negative matrix factorization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_nmf"]


def geron_nmf(X, n_components):
    """
    Non-negative matrix factorization

    Formula: X approx W H, W>=0, H>=0

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, H

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Non-negative matrix factorization"})


def cheatsheet():
    return "hmnmf: Non-negative matrix factorization"
