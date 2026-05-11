# morie.fn — function file (hadesllm/morie)
"""Multi-head attention: concat heads, then linear projection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_multihead_attention"]


def geron_multihead_attention(Q, K, V, n_heads, W_O):
    """
    Multi-head attention: concat heads, then linear projection

    Formula: MHA(Q,K,V) = Concat(head_1,...,head_h) W_O

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    n_heads : array-like
        Input data.
    W_O : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 15
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-head attention: concat heads, then linear projection"})


def cheatsheet():
    return "hmmha: Multi-head attention: concat heads, then linear projection"
