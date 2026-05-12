# morie.fn — function file (hadesllm/morie)
"""Grouped-Query Attention (Ainslie et al. 2023)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["grouped_query_attention"]


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def grouped_query_attention(Q, K=None, V=None, n_heads: int = 8,
                            n_kv_heads: int = 2):
    """Scaled dot-product attention with grouped KV heads.

    ``n_heads`` query heads share ``n_kv_heads`` KV heads; ``n_heads``
    must be a multiple of ``n_kv_heads``.  Following Ainslie et al.
    (2023), each KV head is replicated to ``n_heads / n_kv_heads``
    query heads before computing per-head attention.

    Parameters
    ----------
    Q : ndarray, shape (n_heads, seq_len, d_head)
    K, V : ndarray, shape (n_kv_heads, seq_len, d_head)
    n_heads : int
        Number of query heads.
    n_kv_heads : int
        Number of KV heads.  Must divide n_heads.

    Returns
    -------
    RichResult with keys: tensor (output) shape (n_heads, seq_len, d_head),
    attn (n_heads, seq_len, seq_len).
    """
    if K is None:
        K = Q
    if V is None:
        V = Q
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be a multiple of n_kv_heads")
    group = n_heads // n_kv_heads
    if Q.ndim == 2:
        Q = np.broadcast_to(Q, (n_heads, *Q.shape)).copy()
    if K.ndim == 2:
        K = np.broadcast_to(K, (n_kv_heads, *K.shape)).copy()
    if V.ndim == 2:
        V = np.broadcast_to(V, (n_kv_heads, *V.shape)).copy()
    K_rep = np.repeat(K, group, axis=0)
    V_rep = np.repeat(V, group, axis=0)
    d_head = Q.shape[-1]
    scores = np.einsum("hqd,hkd->hqk", Q, K_rep) / np.sqrt(d_head)
    attn = _softmax(scores, axis=-1)
    out = np.einsum("hqk,hkd->hqd", attn, V_rep)
    return RichResult(
        title="Grouped-Query Attention (Ainslie 2023)",
        summary_lines=[("n_heads", n_heads),
                       ("n_kv_heads", n_kv_heads),
                       ("group_size", group),
                       ("d_head", d_head)],
        payload={"tensor": out, "attn": attn, "n_heads": n_heads,
                 "n_kv_heads": n_kv_heads, "group_size": group,
                 "method": "GQA"},
    )


def cheatsheet():
    return "grpqa(Q,K,V,n_heads,n_kv_heads): grouped-query attention"


# CANONICAL TEST
# >>> Q = np.zeros((4, 2, 8)); K = V = np.zeros((2, 2, 8))
# >>> r = grouped_query_attention(Q, K, V, n_heads=4, n_kv_heads=2)
# >>> r["tensor"].shape
# (4, 2, 8)
