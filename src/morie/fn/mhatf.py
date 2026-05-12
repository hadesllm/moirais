# morie.fn -- function file (hadesllm/morie)
"""Multi-head attention with output projection."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .attnq import scaled_dot_product_attention

__all__ = ["multi_head_attention_full"]


def multi_head_attention_full(x, num_heads: int = 2, W_q=None, W_k=None,
                              W_v=None, W_o=None, seed: int = 0,
                              deterministic_seed: "int | None" = None):
    r"""Multi-head attention with linear projection.

    .. math::

        \\text{MultiHead}(x) =
            \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)\\, W^O

    where each :math:`\\text{head}_i = \\text{Attention}(xW^Q_i,
    xW^K_i, xW^V_i)`.

    Parameters
    ----------
    x : array-like, shape ``(seq_len, d_model)``
        Input sequence.
    num_heads : int
        Number of heads (must divide ``d_model``).
    W_q, W_k, W_v : array-like, shape ``(d_model, d_model)``, optional
        Query/Key/Value projections. Default: small random normal.
    W_o : array-like, shape ``(d_model, d_model)``, optional
        Output projection. Default: identity.
    seed : int
        RNG seed for default projection matrices.
    deterministic_seed : int or None, optional
        If given, the SHA-keyed RNG from
        :func:`morie._det_rng.from_seed` is used so Py<->R streams agree
        for the same ``(name, seed)`` pair. Overrides ``seed`` when set.

    Returns
    -------
    result : RichResult
        Keys: ``output`` / ``estimate``, ``heads`` (attention weights per
        head), ``num_heads``, ``d_k``.

    References
    ----------
    Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by num_heads={num_heads}.")
    d_k = d_model // num_heads

    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("mhatf", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    if W_q is None:
        W_q = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
    if W_k is None:
        W_k = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
    if W_v is None:
        W_v = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
    if W_o is None:
        W_o = np.eye(d_model)
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    W_v = np.asarray(W_v, dtype=float)
    W_o = np.asarray(W_o, dtype=float)

    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v

    Q_h = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    K_h = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V_h = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    head_outputs = []
    head_attns = []
    for h in range(num_heads):
        r = scaled_dot_product_attention(Q_h[h], K_h[h], V_h[h])
        head_outputs.append(r.payload["output"])
        head_attns.append(r.payload["attn"])

    concat = np.concatenate(head_outputs, axis=-1)
    out = concat @ W_o

    return RichResult(
        title=f"Multi-head attention (h={num_heads})",
        summary_lines=[("num_heads", num_heads), ("d_model", d_model),
                       ("d_k per head", d_k), ("seq_len", seq_len)],
        payload={
            "output": out,
            "estimate": out,
            "heads": head_attns,
            "num_heads": int(num_heads),
            "d_k": int(d_k),
            "d_model": int(d_model),
            "method": "Multi-head attention",
        },
    )


# CANONICAL TEST
# x = I_4, h=2 -> output shape (4,4); reproducible given seed=0.


def cheatsheet():
    return "mhatf: MHA = Concat(head_1..head_h) W^O, head_i = Attn(xW_q,...)"
