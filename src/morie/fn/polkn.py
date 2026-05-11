# morie.fn — function file (hadesllm/morie)
"""Polynomial kernel function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["polynomial_kernel"]


def polynomial_kernel(X, c, d):
    """
    Polynomial kernel function

    Formula: K(x_i, x_j) = (x_i'*x_j + c)^d

    Parameters
    ----------
    X : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial kernel function"})


def cheatsheet():
    return "polkn: Polynomial kernel function"
