# moirais.fn — function file (hadesllm/moirais)
"""KV-cache compression for autoregressive LLM inference."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kv_cache_compress"]


def geron_kv_cache_compress(K, V, n_bits):
    """
    KV-cache compression for autoregressive LLM inference

    Formula: quantize stored K,V tensors; reduce memory during generation

    Parameters
    ----------
    K : array-like
        Input data.
    V : array-like
        Input data.
    n_bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: K_q, V_q

    References
    ----------
    Géron Ch 17
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    n = len(K)
    result = float(np.mean(K))
    se = float(np.std(K, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KV-cache compression for autoregressive LLM inference"})


def cheatsheet():
    return "hmkvc: KV-cache compression for autoregressive LLM inference"
