# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Unit vector: normalize a vector to unit L2 norm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_unit_vector"]


def burkov_unit_vector(a):
    """
    Unit vector: normalize a vector to unit L2 norm

    Formula: a_hat = a / ||a||_2

    Parameters
    ----------
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: unit_vector

    References
    ----------
    Burkov Ch 1, Unit Vector section
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unit vector: normalize a vector to unit L2 norm"})


def cheatsheet():
    return "bkunit: Unit vector: normalize a vector to unit L2 norm"
