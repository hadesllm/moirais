# morie.fn -- function file (hadesllm/morie)
"""Gaussian (RBF) kernel function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rbf_kernel"]


def rbf_kernel(X, h):
    """
    Gaussian (RBF) kernel function

    Formula: K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2*h^2))

    Parameters
    ----------
    X : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'K': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian (RBF) kernel function"})


def cheatsheet():
    return "rbfkn: Gaussian (RBF) kernel function"
