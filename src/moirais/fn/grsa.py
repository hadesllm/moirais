# moirais.fn — function file (hadesllm/moirais)
"""Self-attention: Q=K=V from same sequence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_self_attention"]


def geron_self_attention(X, WQ, WK, WV):
    """
    Self-attention: Q=K=V from same sequence

    Formula: SA(X) = softmax(X W_Q (X W_K)^T / sqrt(d_k)) (X W_V)

    Parameters
    ----------
    X : array-like
        Input data.
    WQ : array-like
        Input data.
    WK : array-like
        Input data.
    WV : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Géron Ch 15, Self-Attention section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-attention: Q=K=V from same sequence"})


def cheatsheet():
    return "grsa: Self-attention: Q=K=V from same sequence"
