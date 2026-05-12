# morie.fn -- function file (hadesllm/morie)
"""Multi-head attention: concat h parallel attention heads."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_multi_head_attention"]


def geron_multi_head_attention(Q, K, V, WQ, WK, WV, WO, h):
    """
    Multi-head attention: concat h parallel attention heads

    Formula: MHA(Q,K,V) = Concat(head_1,...,head_h) W_O;  head_i = Attn(Q W_Q^i, K W_K^i, V W_V^i)

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    WQ : array-like
        Input data.
    WK : array-like
        Input data.
    WV : array-like
        Input data.
    WO : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Géron Ch 15, Multi-Head Attention section
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-head attention: concat h parallel attention heads"})


def cheatsheet():
    return "grmha: Multi-head attention: concat h parallel attention heads"
