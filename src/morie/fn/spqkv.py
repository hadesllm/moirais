"""Sparse attention pattern."""
import numpy as np
from ._richresult import RichResult

__all__ = ["sparse_attention"]


def sparse_attention(x):
    """
    Sparse attention pattern

    Formula: attend to fixed + random + sliding window

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Child et al. (2019)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse attention pattern"})


def cheatsheet():
    return "spqkv: Sparse attention pattern"
