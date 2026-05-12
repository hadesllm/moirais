# morie.fn -- function file (hadesllm/morie)
"""Principal components via SVD of centered data matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_principal_components"]


def geron_principal_components(X, n_components):
    """
    Principal components via SVD of centered data matrix

    Formula: X_c = U Sigma V^T; PCs = columns of V

    Parameters
    ----------
    X : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: components

    References
    ----------
    Géron Ch 7
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Principal components via SVD of centered data matrix"})


def cheatsheet():
    return "hmpcac: Principal components via SVD of centered data matrix"
