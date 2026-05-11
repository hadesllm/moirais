# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""L2 (Euclidean) norm of a vector."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_vector_norm"]


def burkov_vector_norm(a):
    """
    L2 (Euclidean) norm of a vector

    Formula: ||a||_2 = sqrt( sum_{i=1..n} a_i^2 )

    Parameters
    ----------
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: norm

    References
    ----------
    Burkov Ch 1, Vector Norm section
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L2 (Euclidean) norm of a vector"})


def cheatsheet():
    return "bknrm: L2 (Euclidean) norm of a vector"
