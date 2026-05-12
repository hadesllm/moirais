# morie.fn -- function file (hadesllm/morie)
"""Scaled dot-product attention (Vaswani et al. 2017)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["scaled_dot_product_attention"]


def _softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K=None, V=None, mask=None):
    r"""Scaled dot-product attention.

    .. math::

        \\text{Attention}(Q, K, V) =
            \\text{softmax}\\!\\left(\\tfrac{QK^\\top}{\\sqrt{d_k}}\\right) V

    Parameters
    ----------
    Q : array-like, shape ``(n_q, d_k)``
        Queries. (If ``K`` and ``V`` are omitted, self-attention is
        performed with ``Q = K = V``.)
    K : array-like, shape ``(n_k, d_k)``, optional
        Keys.
    V : array-like, shape ``(n_k, d_v)``, optional
        Values.
    mask : array-like, shape ``(n_q, n_k)``, optional
        Additive mask (typically 0 / -inf).

    Returns
    -------
    result : RichResult
        Keys: ``output`` / ``estimate``, ``attn`` (attention weights),
        ``logits`` (pre-softmax scores).

    References
    ----------
    Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
    """
    Q = np.asarray(Q, dtype=float)
    if K is None:
        K = Q
    if V is None:
        V = Q
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if Q.ndim == 1:
        Q = Q[None, :]
    if K.ndim == 1:
        K = K[None, :]
    if V.ndim == 1:
        V = V[None, :]
    d_k = K.shape[-1]
    logits = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        logits = logits + np.asarray(mask, dtype=float)
    attn = _softmax(logits, axis=-1)
    out = attn @ V
    return RichResult(
        title="Scaled dot-product attention",
        summary_lines=[("d_k", d_k), ("Q shape", Q.shape),
                       ("K shape", K.shape), ("V shape", V.shape)],
        payload={
            "output": out,
            "estimate": out,
            "attn": attn,
            "logits": logits,
            "d_k": int(d_k),
            "method": "Scaled dot-product attention",
        },
    )


# CANONICAL TEST
# Q = K = V = I_3 (identity), then logits = I/sqrt(3) (diagonal 1/sqrt(3) elsewhere 0)
# softmax of [1/sqrt(3), 0, 0] -> [~0.4548, ~0.2726, ~0.2726]; output is that
# distribution times I = same row.


def cheatsheet():
    return "attnq: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) V"
