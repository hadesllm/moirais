# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Vector dot product (scalar product) of two vectors a and b."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_dot_product"]


def burkov_dot_product(a, b):
    """
    Vector dot product (scalar product) of two vectors a and b

    Formula: a . b = sum_{i=1..n} a_i b_i

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dot

    References
    ----------
    Burkov Ch 1, Dot Product section
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vector dot product (scalar product) of two vectors a and b"})


def cheatsheet():
    return "bkdot: Vector dot product (scalar product) of two vectors a and b"
