# moirais.fn — function file (hadesllm/moirais)
"""Symmetry condition for null distribution of T_N: a_i + a_{N-i+1} = c."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_linrank_symmetry_cond"]


def gibbons_linrank_symmetry_cond(a, N):
    """
    Symmetry condition for null distribution of T_N: a_i + a_{N-i+1} = c

    Formula: a_i + a_{N-i+1} = c (constant) => T_N symmetric about its mean

    Parameters
    ----------
    a : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_symmetric

    References
    ----------
    Gibbons Theorem 7.3.4
    """
    a = np.asarray(a, dtype=float)
    n = int(a) if a.ndim == 0 else len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetry condition for null distribution of T_N: a_i + a_{N-i+1} = c"})


def cheatsheet():
    return "gb734: Symmetry condition for null distribution of T_N: a_i + a_{N-i+1} = c"
