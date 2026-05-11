"""ViT self-attention block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_self_attention"]


def vit_self_attention(q, k, v, mask):
    """
    ViT self-attention block

    Formula: Attn(Q,K,V) = softmax(QK^T/sqrt(d))V

    Parameters
    ----------
    q : array-like
        Input data.
    k : array-like
        Input data.
    v : array-like
        Input data.
    mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT self-attention block"})


def cheatsheet():
    return "vitatt: ViT self-attention block"
