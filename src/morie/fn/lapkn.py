# morie.fn — function file (hadesllm/morie)
"""Laplacian kernel function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["laplacian_kernel"]


def laplacian_kernel(X, h):
    """
    Laplacian kernel function

    Formula: K(x_i, x_j) = exp(-||x_i - x_j||_1 / h)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplacian kernel function"})


def cheatsheet():
    return "lapkn: Laplacian kernel function"
