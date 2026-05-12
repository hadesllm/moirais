# morie.fn — function file (hadesllm/morie)
"""Transformer encoder block (Vaswani et al. 2017)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .mhatf import multi_head_attention_full

__all__ = ["transformer_block"]


def _layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _gelu(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                    (z + 0.044715 * z ** 3)))


def transformer_block(x, num_heads: int = 2, d_ff: "int | None" = None,
                      seed: int = 0):
    """Single Transformer encoder block (post-LN variant).

    .. math::

        h_1 &= \\text{LN}(x + \\text{MHA}(x)) \\\\
        h_2 &= \\text{LN}(h_1 + \\text{FFN}(h_1))

    where ``FFN(z) = GELU(z W_1 + b_1) W_2 + b_2``.

    Parameters
    ----------
    x : array-like, shape ``(seq_len, d_model)``
        Input.
    num_heads : int
        Number of attention heads.
    d_ff : int, optional
        Hidden width of the feed-forward layer. Default ``4 * d_model``.
    seed : int
        RNG seed for default weights.

    Returns
    -------
    result : RichResult
        Keys: ``output`` / ``estimate``, ``h1`` (post-attn LN),
        ``num_heads``.

    References
    ----------
    Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    seq_len, d_model = x.shape
    if d_ff is None:
        d_ff = 4 * d_model

    attn = multi_head_attention_full(x, num_heads=num_heads, seed=seed)
    h1 = _layer_norm(x + attn.payload["output"])

    rng = np.random.default_rng(seed + 1)
    W1 = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_ff))
    b1 = np.zeros(d_ff)
    W2 = rng.normal(0, 1.0 / np.sqrt(d_ff), size=(d_ff, d_model))
    b2 = np.zeros(d_model)

    ffn = _gelu(h1 @ W1 + b1) @ W2 + b2
    h2 = _layer_norm(h1 + ffn)

    return RichResult(
        title=f"Transformer block (h={num_heads}, d_ff={d_ff})",
        summary_lines=[("d_model", d_model), ("num_heads", num_heads),
                       ("d_ff", d_ff), ("seq_len", seq_len)],
        payload={
            "output": h2,
            "estimate": h2,
            "h1": h1,
            "num_heads": int(num_heads),
            "d_ff": int(d_ff),
            "method": "Transformer encoder block (post-LN)",
        },
    )


# CANONICAL TEST
# x = I_4, num_heads=2 -> output shape (4,4); LN rows sum to ~0 and have
# std ~ 1.


def cheatsheet():
    return "trfbl: TF block = LN(x + MHA(x)); LN(h1 + FFN(h1))"
