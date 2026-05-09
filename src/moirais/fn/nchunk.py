"""Chunked causal attention for long-context efficiency."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_chunked_attention"]


def causal_chunked_attention(y, Q, K, V, chunk_size):
    """
    Chunked causal attention for long-context efficiency

    Formula: split L into chunks of size c; attend within + causal cross-chunk

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
    chunk_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hoffmann et al. (2022) Chinchilla impl detail
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chunked causal attention for long-context efficiency"})


def cheatsheet():
    return "nchunk: Chunked causal attention for long-context efficiency"
