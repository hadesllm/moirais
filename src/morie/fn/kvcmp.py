# morie.fn -- function file (hadesllm/morie)
"""KV-cache management and compression for autoregressive decoding.

Two callables share this module:

* ``kv_cache_management`` -- append a new (k, v) pair to a running KV
  cache (Pope et al. 2022).  This is the canonical morie.fn spec entry
  ``kvcmp``.
* ``kv_cache_compress``   -- TurboQuant-style quantised compression of
  an existing (K, V) cache (kept for backward-compatible imports).
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult

__all__ = ["kv_cache_management", "kv_cache_compress"]


# ───────────────────────── canonical spec API ──────────────────────────

def kv_cache_management(K_cache, V_cache, k_new, v_new,
                        max_len: int | None = None) -> RichResult:
    """Append a new (k, v) pair to a running KV cache.

    Formula:
        K_new = concat(K_cache, k_t),  V_new = concat(V_cache, v_t)

    If ``max_len`` is given, the cache is right-trimmed to its last
    ``max_len`` timesteps (sliding-window cache).

    Parameters
    ----------
    K_cache, V_cache : ndarray, shape (..., T_cache, d_head) or None
    k_new, v_new : ndarray, shape (..., T_new, d_head) or (..., d_head)
    max_len : int, optional

    Returns
    -------
    RichResult with keys: K (new cache), V (new cache), T (new length).
    """
    k_new = np.asarray(k_new, dtype=float)
    v_new = np.asarray(v_new, dtype=float)
    if K_cache is None:
        K_new = k_new if k_new.ndim >= 2 else np.expand_dims(k_new, axis=-2)
        V_new = v_new if v_new.ndim >= 2 else np.expand_dims(v_new, axis=-2)
    else:
        K_cache = np.asarray(K_cache, dtype=float)
        V_cache = np.asarray(V_cache, dtype=float)
        if k_new.ndim == K_cache.ndim - 1:
            k_new = np.expand_dims(k_new, axis=-2)
            v_new = np.expand_dims(v_new, axis=-2)
        K_new = np.concatenate([K_cache, k_new], axis=-2)
        V_new = np.concatenate([V_cache, v_new], axis=-2)
    if max_len is not None and K_new.shape[-2] > max_len:
        K_new = K_new[..., -max_len:, :]
        V_new = V_new[..., -max_len:, :]
    T = K_new.shape[-2]
    return RichResult(
        title="KV-Cache Append (Pope 2022)",
        summary_lines=[("T_new", T),
                       ("d_head", K_new.shape[-1])],
        payload={"K": K_new, "V": V_new, "T": T,
                 "max_len": max_len, "method": "kv-cache-append"},
    )


def cheatsheet() -> str:
    return ("kvcmp(K_cache, V_cache, k_new, v_new, max_len): KV-cache append; "
            "kv_cache_compress(K, V, bits): TurboQuant quantisation")


# CANONICAL TEST
# >>> K, V = np.zeros((2, 4)), np.zeros((2, 4))
# >>> r = kv_cache_management(K, V, np.ones((1, 4)), np.ones((1, 4)))
# >>> r["T"]
# 3


# ─────────────────── legacy TurboQuant quantisation API ────────────────

def kv_cache_compress(
    K: np.ndarray,
    V: np.ndarray,
    bits: int = 3,
) -> DescriptiveResult:
    """Compress KV-cache matrices using TurboQuant-style quantization.

    Applies per-head uniform quantization to both K and V.

    :param K: Key matrix (seq_len x d_head) or (n_heads x seq_len x d_head).
    :param V: Value matrix (same shape as K).
    :param bits: Quantization bit width.
    :return: DescriptiveResult with compression ratio and cosine similarity.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if K.ndim == 2:
        K = K[None, :, :]
        V = V[None, :, :]
    n_heads, seq_len, d_head = K.shape
    levels = 2**bits

    def _quantize_vec(vec):
        vmin, vmax = vec.min(), vec.max()
        if vmax > vmin:
            codes = np.clip(
                np.round((vec - vmin) / (vmax - vmin) * (levels - 1)),
                0,
                levels - 1,
            )
            return codes / (levels - 1) * (vmax - vmin) + vmin
        return vec.copy()

    total_cosine_k = 0.0
    total_cosine_v = 0.0
    count = n_heads * seq_len
    for h in range(n_heads):
        for s in range(seq_len):
            k_vec = K[h, s]
            k_hat = _quantize_vec(k_vec)
            nk, nkh = np.linalg.norm(k_vec), np.linalg.norm(k_hat)
            total_cosine_k += (np.dot(k_vec, k_hat) / (nk * nkh)) if nk > 0 and nkh > 0 else 1.0

            v_vec = V[h, s]
            v_hat = _quantize_vec(v_vec)
            nv, nvh = np.linalg.norm(v_vec), np.linalg.norm(v_hat)
            total_cosine_v += (np.dot(v_vec, v_hat) / (nv * nvh)) if nv > 0 and nvh > 0 else 1.0

    avg_cosine = float((total_cosine_k + total_cosine_v) / (2 * count))
    ratio = 32.0 / bits
    return DescriptiveResult(
        name="kv_cache_compress",
        value=avg_cosine,
        extra={
            "compression_ratio": ratio,
            "cosine_similarity": avg_cosine,
            "bits": bits,
            "shape": (n_heads, seq_len, d_head),
            "cosine_K": float(total_cosine_k / count),
            "cosine_V": float(total_cosine_v / count),
        },
    )


kvcmp = kv_cache_management
