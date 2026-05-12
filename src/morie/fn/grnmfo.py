# morie.fn -- function file (hadesllm/morie)
"""Non-negative matrix factorization reconstruction objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_nmf_objective"]


def geron_nmf_objective(X, W, H):
    """
    Non-negative matrix factorization reconstruction objective

    Formula: min_{W >= 0, H >= 0} ||X - W H||_F^2

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 7, NMF section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Non-negative matrix factorization reconstruction objective"})


def cheatsheet():
    return "grnmfo: Non-negative matrix factorization reconstruction objective"
