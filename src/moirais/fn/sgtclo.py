"""Closeness centrality (Wasserman-Faust)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_closeness_centrality"]


def sgt_closeness_centrality(A):
    """
    Closeness centrality (Wasserman-Faust)

    Formula: C(v) = (n-1)/Σ_u d(v,u)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C

    References
    ----------
    Wasserman-Faust (1994)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closeness centrality (Wasserman-Faust)"})


def cheatsheet():
    return "sgtclo: Closeness centrality (Wasserman-Faust)"
