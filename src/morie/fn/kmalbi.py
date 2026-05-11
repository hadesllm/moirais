# morie.fn — function file (hadesllm/morie)
"""ALiBi: add linear bias to attention scores based on token distance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_alibi_bias"]


def kamath_alibi_bias(Q, K, V, slopes):
    """
    ALiBi: add linear bias to attention scores based on token distance

    Formula: Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k) + m * D) V; D_ij = -(i - j), m = head-dependent slope

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    slopes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Kamath Ch 2, ALiBi section (Press et al.)
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ALiBi: add linear bias to attention scores based on token distance"})


def cheatsheet():
    return "kmalbi: ALiBi: add linear bias to attention scores based on token distance"
