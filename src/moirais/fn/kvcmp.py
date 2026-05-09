# moirais.fn — function file (hadesllm/moirais)
"""KV-cache compression pipeline."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


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


def cheatsheet() -> str:
    return "kv_cache_compress(K, V, bits) -> KV-cache compression pipeline"


kvcmp = kv_cache_compress
