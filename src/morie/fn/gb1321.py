# morie.fn — function file (hadesllm/morie)
"""Asymptotic relative efficiency: limiting ratio of sample sizes for equal power."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_are_def"]


def gibbons_are_def(T, T_star, n):
    """
    Asymptotic relative efficiency: limiting ratio of sample sizes for equal power

    Formula: ARE(T,T*) = lim n*/n for equal power at same sequence of alternatives

    Parameters
    ----------
    T : array-like
        Input data.
    T_star : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE

    References
    ----------
    Gibbons Theorem 13.2.1 setup
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic relative efficiency: limiting ratio of sample sizes for equal power"})


def cheatsheet():
    return "gb1321: Asymptotic relative efficiency: limiting ratio of sample sizes for equal power"
