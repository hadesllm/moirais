# morie.fn -- function file (hadesllm/morie)
"""Gradient clipping by norm."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gradient_clip(
    grads: list[np.ndarray],
    max_norm: float = 1.0,
) -> DescriptiveResult:
    """Clip gradients by global norm.

    If the global L2 norm of all gradients exceeds *max_norm*, each gradient
    tensor is scaled down proportionally.

    :param grads: List of gradient arrays.
    :param max_norm: Maximum allowed global norm.
    :return: DescriptiveResult with clipped gradients in ``extra['clipped']``.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")

    total_norm = np.sqrt(sum(float(np.sum(g**2)) for g in grads))
    clip_coef = max_norm / max(total_norm, 1e-12)
    if clip_coef < 1.0:
        clipped = [g * clip_coef for g in grads]
    else:
        clipped = [g.copy() for g in grads]

    return DescriptiveResult(
        name="gradient_clip",
        value=float(total_norm),
        extra={
            "clipped": clipped,
            "max_norm": max_norm,
            "was_clipped": clip_coef < 1.0,
            "clip_coef": float(min(clip_coef, 1.0)),
        },
    )


def cheatsheet() -> str:
    return "gradient_clip(grads, max_norm) -> clip gradients by global norm"


gradc = gradient_clip
