# morie.fn — function file (hadesllm/morie)
"""QJL random projection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qjl_project(
    x: np.ndarray,
    d_target: int = 64,
    bits: int = 1,
    seed: int = 42,
) -> DescriptiveResult:
    """Quantized Johnson-Lindenstrauss random projection.

    Projects *x* from d dimensions to *d_target* using a random
    Rademacher matrix, then quantizes to *bits* per entry.

    :param x: Input vector (1-D or 2-D).
    :param d_target: Target dimensionality.
    :param bits: Bits per projected entry (1 = sign only).
    :param seed: Random seed for reproducibility.
    :return: DescriptiveResult with projected data.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    d_in = x.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.choice([-1.0, 1.0], size=(d_in, d_target)) / np.sqrt(d_target)
    projected = x @ R
    if bits == 1:
        quantized = np.sign(projected).astype(np.int8)
        quantized[quantized == 0] = 1
    else:
        vmin, vmax = projected.min(), projected.max()
        levels = 2**bits
        if vmax > vmin:
            quantized = np.clip(
                np.round((projected - vmin) / (vmax - vmin) * (levels - 1)),
                0,
                levels - 1,
            ).astype(np.int32)
        else:
            quantized = np.zeros_like(projected, dtype=np.int32)
    return DescriptiveResult(
        name="qjl_project",
        value=float(d_target),
        extra={
            "projected": projected.squeeze(),
            "quantized": quantized.squeeze(),
            "d_in": d_in,
            "d_target": d_target,
            "bits": bits,
            "compression_ratio": (d_in * 32) / (d_target * bits),
        },
    )


def cheatsheet() -> str:
    return "qjl_project(x, d_target, bits=1) -> QJL random projection"


qjl = qjl_project
