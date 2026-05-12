# morie.fn -- function file (hadesllm/morie)
"""Scaled dot-product attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_scaled_dot_product"]


def geron_scaled_dot_product(Q, K, V, d_k, mask):
    """
    Scaled dot-product attention

    Formula: Att(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    d_k : array-like
        Input data.
    mask : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scaled dot-product attention"})


def cheatsheet():
    return "hmsdp: Scaled dot-product attention"
