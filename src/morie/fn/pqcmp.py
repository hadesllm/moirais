# morie.fn — function file (hadesllm/morie)
"""PolarQuant full compression pipeline."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polar_compress(
    x: np.ndarray,
    bits_mag: int = 8,
    bits_dir: int = 4,
) -> DescriptiveResult:
    """Full PolarQuant pipeline: normalize, quantize magnitude + direction.

    :param x: Input vector.
    :param bits_mag: Bits for magnitude scalar.
    :param bits_dir: Bits per direction component.
    :return: DescriptiveResult with compressed representation.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    d = len(x)
    magnitude = float(np.linalg.norm(x))
    if magnitude > 0:
        direction = x / magnitude
    else:
        direction = np.zeros_like(x)
    levels_dir = 2**bits_dir
    dir_min, dir_max = float(direction.min()), float(direction.max())
    if dir_max > dir_min:
        codes = np.clip(
            np.round((direction - dir_min) / (dir_max - dir_min) * (levels_dir - 1)),
            0,
            levels_dir - 1,
        ).astype(np.int32)
        dir_hat = codes.astype(np.float64) / (levels_dir - 1) * (dir_max - dir_min) + dir_min
    else:
        codes = np.zeros(d, dtype=np.int32)
        dir_hat = direction.copy()
    reconstructed = magnitude * dir_hat
    mse = float(np.mean((x - reconstructed) ** 2))
    total_bits = bits_mag + d * bits_dir
    ratio = (d * 32) / total_bits
    return DescriptiveResult(
        name="polar_compress",
        value=mse,
        extra={
            "magnitude": magnitude,
            "dir_codes": codes,
            "dir_range": (dir_min, dir_max),
            "reconstructed": reconstructed,
            "compression_ratio": ratio,
            "bits_mag": bits_mag,
            "bits_dir": bits_dir,
        },
    )


def cheatsheet() -> str:
    return "polar_compress(x, bits_mag, bits_dir) -> full PQ pipeline"


pqcmp = polar_compress
