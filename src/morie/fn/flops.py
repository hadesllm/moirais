# morie.fn -- function file (hadesllm/morie)
"""FLOPs estimator for transformer models."""

from __future__ import annotations

from ._containers import DescriptiveResult


def estimate_flops(
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int = 32000,
    batch_size: int = 1,
) -> DescriptiveResult:
    r"""Estimate FLOPs for a single forward pass of a transformer.

    Uses the Kaplan et al. approximation:
    :math:`\\text{FLOPs} \\approx 2 \\cdot N \\cdot T` for a model with
    *N* parameters and sequence length *T*, plus attention cost.

    :param seq_len: Sequence length (T).
    :param d_model: Model dimension.
    :param n_layers: Number of transformer layers.
    :param n_heads: Number of attention heads.
    :param vocab_size: Vocabulary size.
    :param batch_size: Batch size.
    :return: DescriptiveResult with FLOPs breakdown.
    """
    d_head = d_model // n_heads

    qkv_flops = 3 * 2 * seq_len * d_model * d_model
    attn_flops = 2 * seq_len * seq_len * d_model
    proj_flops = 2 * seq_len * d_model * d_model
    ffn_flops = 2 * 2 * seq_len * d_model * (4 * d_model)

    per_layer = qkv_flops + attn_flops + proj_flops + ffn_flops
    embed_flops = 2 * seq_len * vocab_size * d_model
    total = (n_layers * per_layer + embed_flops) * batch_size

    return DescriptiveResult(
        name="estimate_flops",
        value=total,
        extra={
            "total_flops": total,
            "per_layer": per_layer * batch_size,
            "attention_flops": (qkv_flops + attn_flops + proj_flops) * n_layers * batch_size,
            "ffn_flops": ffn_flops * n_layers * batch_size,
            "embed_flops": embed_flops * batch_size,
            "total_tflops": total / 1e12,
            "n_layers": n_layers,
            "d_head": d_head,
        },
    )


def cheatsheet() -> str:
    return "estimate_flops(seq_len, d_model, n_layers, n_heads) -> FLOPs estimate"


flops = estimate_flops
