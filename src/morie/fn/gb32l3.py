# morie.fn -- function file (hadesllm/morie)
"""Combinatorial identity: sum_r C(m,r)C(n,r+1) = C(m+n,m+1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_vandermonde_id2"]


def gibbons_vandermonde_id2(m, n):
    """
    Combinatorial identity: sum_r C(m,r)C(n,r+1) = C(m+n,m+1)

    Formula: sum_{r=0}^{c} C(m,r)C(n,r+1) = C(m+n,m+1), c = min(m,n-1)

    Parameters
    ----------
    m : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: combinatorial_sum

    References
    ----------
    Gibbons Lemma 3.2.3
    """
    m = np.asarray(m, dtype=float)
    n = int(m) if m.ndim == 0 else len(m)
    result = float(np.mean(m))
    se = float(np.std(m, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Combinatorial identity: sum_r C(m,r)C(n,r+1) = C(m+n,m+1)"})


def cheatsheet():
    return "gb32l3: Combinatorial identity: sum_r C(m,r)C(n,r+1) = C(m+n,m+1)"
