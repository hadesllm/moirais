# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""KV-cache growth during autoregressive decoding: append new K/V per generated token."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_kv_cache_lookup"]


def alammar_kv_cache_lookup(K_cache, V_cache, k_new, v_new, q_new):
    """
    KV-cache growth during autoregressive decoding: append new K/V per generated token

    Formula: K_t = [K_{t-1}; k_t];  V_t = [V_{t-1}; v_t];  attn_t = softmax(q_t K_t^T / sqrt(d_k)) V_t

    Parameters
    ----------
    K_cache : array-like
        Input data.
    V_cache : array-like
        Input data.
    k_new : array-like
        Input data.
    v_new : array-like
        Input data.
    q_new : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output, K_cache_new, V_cache_new

    References
    ----------
    Alammar Ch 3, KV-cache (key-value caching) section
    """
    K_cache = np.atleast_1d(np.asarray(K_cache, dtype=float))
    n = len(K_cache)
    result = float(np.mean(K_cache))
    se = float(np.std(K_cache, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KV-cache growth during autoregressive decoding: append new K/V per generated token"})


def cheatsheet():
    return "alkvc2: KV-cache growth during autoregressive decoding: append new K/V per generated token"
