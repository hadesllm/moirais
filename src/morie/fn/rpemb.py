# morie.fn — function file (hadesllm/morie)
"""Rotary Positional Embedding (RoPE)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rotary_embed(
    x: np.ndarray,
    seq_len: int | None = None,
    dim: int | None = None,
    base: float = 10000.0,
) -> DescriptiveResult:
    """Apply Rotary Positional Embedding to input.

    :math:`\\text{RoPE}(x, m) = x \\odot \\cos(m\\theta) + \\text{rotate}(x) \\odot \\sin(m\\theta)`

    where :math:`\\theta_i = \\text{base}^{-2i/d}`.

    :param x: Input array of shape (seq_len, dim) or (batch, seq_len, dim).
    :param seq_len: Sequence length (inferred from x if None).
    :param dim: Embedding dimension (inferred from x if None).
    :param base: Base for frequency computation.
    :return: DescriptiveResult with embedded output in ``extra['output']``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        s, d = x.shape
    elif x.ndim == 3:
        _, s, d = x.shape
    else:
        raise ValueError(f"x must be 2D or 3D, got {x.ndim}D")

    if seq_len is None:
        seq_len = s
    if dim is None:
        dim = d

    half_dim = dim // 2
    freqs = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float64) / half_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, freqs)

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    if x.ndim == 2:
        x1, x2 = x[:, :half_dim], x[:, half_dim : 2 * half_dim]
        out1 = x1 * cos_vals[:seq_len] - x2 * sin_vals[:seq_len]
        out2 = x1 * sin_vals[:seq_len] + x2 * cos_vals[:seq_len]
        output = x.copy()
        output[:, :half_dim] = out1
        output[:, half_dim : 2 * half_dim] = out2
    else:
        x1 = x[:, :, :half_dim]
        x2 = x[:, :, half_dim : 2 * half_dim]
        c = cos_vals[np.newaxis, :seq_len, :]
        s_ = sin_vals[np.newaxis, :seq_len, :]
        out1 = x1 * c - x2 * s_
        out2 = x1 * s_ + x2 * c
        output = x.copy()
        output[:, :, :half_dim] = out1
        output[:, :, half_dim : 2 * half_dim] = out2

    return DescriptiveResult(
        name="rotary_embed",
        value=float(base),
        extra={"output": output, "seq_len": seq_len, "dim": dim, "base": base},
    )


def cheatsheet() -> str:
    return "rotary_embed(x, seq_len, dim) -> RoPE positional encoding"


rpemb = rotary_embed
