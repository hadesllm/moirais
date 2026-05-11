"""Uniform scalar quantizer."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def uniform_quantize(
    x: np.ndarray,
    bits: int = 8,
    vmin: float | None = None,
    vmax: float | None = None,
) -> DescriptiveResult:
    """Uniform (linear) scalar quantization.

    Maps values linearly into 2^bits bins between vmin and vmax.

    :param x: Input data.
    :param bits: Quantization bit width.
    :param vmin: Minimum clip value (default: data min).
    :param vmax: Maximum clip value (default: data max).
    :return: DescriptiveResult with quantized values and MSE.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if vmin is None:
        vmin = float(x.min())
    if vmax is None:
        vmax = float(x.max())
    levels = 2**bits
    if vmax <= vmin:
        codes = np.zeros(len(x), dtype=np.int32)
        x_hat = np.full_like(x, vmin)
    else:
        x_clip = np.clip(x, vmin, vmax)
        codes = np.clip(
            np.floor((x_clip - vmin) / (vmax - vmin) * levels).astype(np.int32),
            0,
            levels - 1,
        )
        step = (vmax - vmin) / levels
        x_hat = codes.astype(np.float64) * step + vmin + step / 2
    mse = float(np.mean((x - x_hat) ** 2))
    return DescriptiveResult(
        name="uniform_quantize",
        value=mse,
        extra={
            "codes": codes,
            "reconstructed": x_hat,
            "bits": bits,
            "levels": levels,
            "vmin": vmin,
            "vmax": vmax,
            "mse": mse,
        },
    )


def cheatsheet() -> str:
    return "uniform_quantize(x, bits, vmin, vmax) -> uniform quantization"


unfrm = uniform_quantize
