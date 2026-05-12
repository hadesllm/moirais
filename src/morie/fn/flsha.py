# morie.fn -- function file (hadesllm/morie)
"""FlashAttention (IO-aware tiled softmax; Dao et al. 2022)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["flash_attention"]


def flash_attention(Q, K=None, V=None, block_size: int = 32,
                    mask=None):
    """Tiled scaled-dot-product attention with the online-softmax
    recurrence used by FlashAttention.

    The output is numerically identical (within FP roundoff) to the
    naive ``softmax(QK^T / sqrt(d)) V``, but it is computed in tiles
    so that the full (N x N) attention matrix never materialises.
    This is the IO-aware algorithm of Dao et al. (2022).

    Parameters
    ----------
    Q, K, V : ndarray, shape (N, d)
        Single-head attention inputs.
    block_size : int
        KV tile size; output is mathematically invariant to it.
    mask : ndarray (N, N), optional
        Additive attention mask (e.g. -inf for causal).

    Returns
    -------
    RichResult with keys: tensor (output), block_size.
    """
    Q = np.asarray(Q, dtype=float)
    K = Q if K is None else np.asarray(K, dtype=float)
    V = Q if V is None else np.asarray(V, dtype=float)
    N, d = Q.shape
    M = K.shape[0]
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(Q)
    row_max = np.full((N,), -np.inf)
    row_denom = np.zeros((N,))
    for j_start in range(0, M, block_size):
        j_end = min(j_start + block_size, M)
        Kj = K[j_start:j_end]
        Vj = V[j_start:j_end]
        s = (Q @ Kj.T) * scale
        if mask is not None:
            s = s + np.asarray(mask, dtype=float)[:, j_start:j_end]
        block_max = np.max(s, axis=1)
        new_max = np.maximum(row_max, block_max)
        alpha = np.exp(row_max - new_max)
        beta = np.exp(s - new_max[:, None])
        row_denom = row_denom * alpha + np.sum(beta, axis=1)
        out = out * alpha[:, None] + beta @ Vj
        row_max = new_max
    out = out / row_denom[:, None]
    return RichResult(
        title="FlashAttention (Dao 2022)",
        summary_lines=[("N", N), ("d", d), ("block_size", block_size)],
        payload={"tensor": out, "block_size": block_size,
                 "method": "flash-attention"},
    )


def cheatsheet():
    return "flshA(Q, K, V, block_size): IO-aware tiled attention"


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> Q = rng.standard_normal((6, 4)); K = rng.standard_normal((6, 4))
# >>> V = rng.standard_normal((6, 4))
# >>> r = flash_attention(Q, K, V, block_size=2)
# >>> s = Q @ K.T / np.sqrt(4)
# >>> p = np.exp(s - s.max(1, keepdims=True))
# >>> p = p / p.sum(1, keepdims=True)
# >>> bool(np.allclose(r["tensor"], p @ V, atol=1e-10))
# True
