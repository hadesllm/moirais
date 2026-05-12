# morie.fn — function file (hadesllm/morie)
"""RMS normalization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rms_norm(
    x: np.ndarray,
    eps: float = 1e-6,
    weight: np.ndarray | None = None,
) -> DescriptiveResult:
    r"""Apply Root Mean Square Layer Normalization.

    :math:`\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}} \\odot \\gamma`

    Used in Llama, Gemma, and other modern transformer architectures.

    :param x: Input array of shape (..., d).
    :param eps: Epsilon for numerical stability.
    :param weight: Optional learnable scale parameter of shape (d,).
    :return: DescriptiveResult with normalized array in ``extra['output']``.
    """
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    normed = x / rms
    if weight is not None:
        normed = normed * weight

    return DescriptiveResult(
        name="rms_norm",
        value=float(np.mean(rms)),
        extra={"output": normed, "rms_mean": float(np.mean(rms)), "eps": eps},
    )


def cheatsheet() -> str:
    return "rms_norm(x, eps) -> RMS layer normalization"


rmsnm = rms_norm
