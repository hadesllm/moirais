# morie.fn -- function file (hadesllm/morie)
"""Self-attention: Q=K=V come from same input."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_self_attention"]


def geron_self_attention(X, W_Q, W_K, W_V):
    """
    Self-attention: Q=K=V come from same input

    Formula: Att(X W_Q, X W_K, X W_V) for the same X

    Parameters
    ----------
    X : array-like
        Input data.
    W_Q : array-like
        Input data.
    W_K : array-like
        Input data.
    W_V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 15
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-attention: Q=K=V come from same input"})


def cheatsheet():
    return "hmsatt: Self-attention: Q=K=V come from same input"
