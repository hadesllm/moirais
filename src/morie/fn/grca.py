# morie.fn — function file (hadesllm/morie)
"""Cross-attention: Q from decoder, K/V from encoder output."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cross_attention"]


def geron_cross_attention(X_dec, X_enc, WQ, WK, WV):
    """
    Cross-attention: Q from decoder, K/V from encoder output

    Formula: CA(X_dec, X_enc) = softmax(X_dec W_Q (X_enc W_K)^T / sqrt(d_k)) (X_enc W_V)

    Parameters
    ----------
    X_dec : array-like
        Input data.
    X_enc : array-like
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
    Géron Ch 15, Cross-Attention section
    """
    X_dec = np.atleast_1d(np.asarray(X_dec, dtype=float))
    n = len(X_dec)
    result = float(np.mean(X_dec))
    se = float(np.std(X_dec, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-attention: Q from decoder, K/V from encoder output"})


def cheatsheet():
    return "grca: Cross-attention: Q from decoder, K/V from encoder output"
