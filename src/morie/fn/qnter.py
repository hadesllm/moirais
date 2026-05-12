# morie.fn -- function file (hadesllm/morie)
"""Quantization error analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Errors using inadequate data are much less than those using no data at all. -- Charles Babbage"


def quantization_error(x, bits: int = 8) -> DescriptiveResult:
    """Compute quantization noise e = x - Q(x).

    Parameters
    ----------
    x : array-like
        Input signal.
    bits : int
        Number of quantization bits. Default 8.

    Returns
    -------
    DescriptiveResult
    """
    from morie.fn.sqntz import quantize_signal

    x = np.asarray(x, dtype=float)
    qr = quantize_signal(x, bits=bits)
    error = x - qr.filtered
    mse = float(np.mean(error**2))
    snr = 10.0 * np.log10(np.mean(x**2) / mse) if mse > 0 else float("inf")
    return DescriptiveResult(
        name="quantization_error",
        value=mse,
        extra={"error": error, "mse": mse, "snr_db": float(snr), "bits": bits},
    )


qnter = quantization_error


def cheatsheet() -> str:
    return "quantization_error({}) -> Quantization error analysis."
