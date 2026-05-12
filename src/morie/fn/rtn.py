# morie.fn -- function file (hadesllm/morie)
"""Round-to-nearest (RTN) baseline quantizer."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def round_to_nearest(
    x: np.ndarray,
    bits: int = 4,
) -> DescriptiveResult:
    """Round-to-nearest quantization baseline.

    The simplest quantization scheme: scale to [0, 2^bits - 1],
    round, and scale back.

    :param x: Input data.
    :param bits: Quantization bit width.
    :return: DescriptiveResult with RTN-quantized values and MSE.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    vmin, vmax = float(x.min()), float(x.max())
    levels = 2**bits - 1
    if vmax <= vmin:
        x_hat = np.full_like(x, vmin)
        codes = np.zeros(len(x), dtype=np.int32)
    else:
        scaled = (x - vmin) / (vmax - vmin) * levels
        codes = np.clip(np.round(scaled), 0, levels).astype(np.int32)
        x_hat = codes.astype(np.float64) / levels * (vmax - vmin) + vmin
    mse = float(np.mean((x - x_hat) ** 2))
    return DescriptiveResult(
        name="round_to_nearest",
        value=mse,
        extra={
            "codes": codes,
            "reconstructed": x_hat,
            "bits": bits,
            "vmin": vmin,
            "vmax": vmax,
            "mse": mse,
        },
    )


def cheatsheet() -> str:
    return "round_to_nearest(x, bits) -> RTN baseline quantizer"


rtn = round_to_nearest
