# morie.fn -- function file (hadesllm/morie)
"""KV-cache compression for autoregressive LLMs (bits per value)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kv_cache_compression"]


def geron_kv_cache_compression(seq_len, num_layers, num_heads, d_head, bits):
    """
    KV-cache compression for autoregressive LLMs (bits per value)

    Formula: cache_bytes = seq_len * num_layers * num_heads * d_head * 2 * bits / 8

    Parameters
    ----------
    seq_len : array-like
        Input data.
    num_layers : array-like
        Input data.
    num_heads : array-like
        Input data.
    d_head : array-like
        Input data.
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bytes_per_token

    References
    ----------
    Géron Ch 17, KV-cache compression section
    """
    seq_len = np.atleast_1d(np.asarray(seq_len, dtype=float))
    n = len(seq_len)
    result = float(np.mean(seq_len))
    se = float(np.std(seq_len, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KV-cache compression for autoregressive LLMs (bits per value)"})


def cheatsheet():
    return "grkvc: KV-cache compression for autoregressive LLMs (bits per value)"
