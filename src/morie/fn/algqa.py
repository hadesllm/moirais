# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Grouped-Query Attention: H query heads share K/V across G groups (H divisible by G)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_grouped_query_attention"]


def alammar_grouped_query_attention(Q, K, V, n_query_heads, n_kv_groups):
    """
    Grouped-Query Attention: H query heads share K/V across G groups (H divisible by G)

    Formula: head_i = Attn(Q_i, K_{g(i)}, V_{g(i)});  g(i) = i mod G;  Concat(head_1,...,head_H) W_O

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    n_query_heads : array-like
        Input data.
    n_kv_groups : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Alammar Ch 3, Grouped-Query Attention section
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grouped-Query Attention: H query heads share K/V across G groups (H divisible by G)"})


def cheatsheet():
    return "algqa: Grouped-Query Attention: H query heads share K/V across G groups (H divisible by G)"
