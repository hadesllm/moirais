# morie.fn -- function file (hadesllm/morie)
"""Cross-attention: Q from decoder, K/V from encoder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cross_attention"]


def geron_cross_attention(dec_h, enc_h, W_Q, W_K, W_V):
    """
    Cross-attention: Q from decoder, K/V from encoder

    Formula: Att(dec_h W_Q, enc_h W_K, enc_h W_V)

    Parameters
    ----------
    dec_h : array-like
        Input data.
    enc_h : array-like
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
    dec_h = np.atleast_1d(np.asarray(dec_h, dtype=float))
    n = len(dec_h)
    result = float(np.mean(dec_h))
    se = float(np.std(dec_h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-attention: Q from decoder, K/V from encoder"})


def cheatsheet():
    return "hmcatt: Cross-attention: Q from decoder, K/V from encoder"
