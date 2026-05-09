# moirais.fn — function file (hadesllm/moirais)
"""Special case symmetry when N even and a_i = i for i <= N/2, else N-i+1."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_linrank_sym_special"]


def gibbons_linrank_sym_special(N):
    """
    Special case symmetry when N even and a_i = i for i <= N/2, else N-i+1

    Formula: a_i = i (i<=N/2), a_i = N-i+1 (i>N/2) => T_N symmetric

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores

    References
    ----------
    Gibbons Theorem 7.3.6
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Special case symmetry when N even and a_i = i for i <= N/2, else N-i+1"})


def cheatsheet():
    return "gb736: Special case symmetry when N even and a_i = i for i <= N/2, else N-i+1"
