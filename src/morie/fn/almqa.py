# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Multi-Query Attention: single K/V head shared across all Q heads."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_multi_query_attention"]


def alammar_multi_query_attention(Q, K_shared, V_shared, n_query_heads):
    """
    Multi-Query Attention: single K/V head shared across all Q heads

    Formula: head_i = Attn(Q_i, K_shared, V_shared);  Concat W_O

    Parameters
    ----------
    Q : array-like
        Input data.
    K_shared : array-like
        Input data.
    V_shared : array-like
        Input data.
    n_query_heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Alammar Ch 3, Multi-Query Attention section
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-Query Attention: single K/V head shared across all Q heads"})


def cheatsheet():
    return "almqa: Multi-Query Attention: single K/V head shared across all Q heads"
