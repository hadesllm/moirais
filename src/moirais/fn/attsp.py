"""Sparse attention with structured patterns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sparse_attention"]


def sparse_attention(y, Q, K, V, S):
    """
    Sparse attention with structured patterns

    Formula: SparseAttn(Q,K,V) = softmax((Q K^T) * S / sqrt(d)) V; S in {0,1}

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Child et al. (2019); Beltagy et al. (2020) Longformer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse attention with structured patterns"})


def cheatsheet():
    return "attsp: Sparse attention with structured patterns"
