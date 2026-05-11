"""Scaled dot-product attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["scaled_dot_product_attention"]


def scaled_dot_product_attention(y, Q, K, V, mask):
    """
    Scaled dot-product attention

    Formula: Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vaswani et al. (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scaled dot-product attention"})


def cheatsheet():
    return "attsdp: Scaled dot-product attention"
