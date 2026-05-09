# moirais.fn — function file (hadesllm/moirais)
"""Continuous wavelet transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cwt_compute_fn(
    x: np.ndarray,
    scales: np.ndarray | None = None,
    wavelet: str = "morlet",
    fs: float = 1.0,
) -> DescriptiveResult:
    """Compute continuous wavelet transform of a signal.

    :param x: 1-D input signal.
    :param scales: Array of scales; auto-generated if None.
    :param wavelet: Wavelet type, 'morlet' or 'mexican_hat' (default 'morlet').
    :param fs: Sampling frequency in Hz (default 1.0).
    :return: DescriptiveResult with CWT coefficients, scales, and frequencies.
    """
    from moirais._adaptive import cwt_compute

    x = np.asarray(x, dtype=float).ravel()
    coeffs, scales_out, freqs = cwt_compute(x, scales=scales, wavelet=wavelet, fs=fs)
    return DescriptiveResult(
        name="cwt",
        value=len(scales_out),
        extra={"coefficients": coeffs, "scales": scales_out, "frequencies": freqs},
    )


cwtfn = cwt_compute_fn


def cheatsheet() -> str:
    return "cwt_compute_fn({}) -> Continuous wavelet transform."
