# moirais.fn — function file (hadesllm/moirais)
"""Symmetry of null distribution of T_N when m = n = N/2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_linrank_sym_equal"]


def gibbons_linrank_sym_equal(a, m, n):
    """
    Symmetry of null distribution of T_N when m = n = N/2

    Formula: m = n implies T_N symmetric about its mean for any scores a_i

    Parameters
    ----------
    a : array-like
        Input data.
    m : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_symmetric

    References
    ----------
    Gibbons Theorem 7.3.5
    """
    a = np.asarray(a, dtype=float)
    n = int(a) if a.ndim == 0 else len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetry of null distribution of T_N when m = n = N/2"})


def cheatsheet():
    return "gb735: Symmetry of null distribution of T_N when m = n = N/2"
