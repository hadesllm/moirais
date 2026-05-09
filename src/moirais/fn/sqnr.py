"""Signal-to-quantization-noise ratio (SQNR)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def signal_quant_noise_ratio(
    x: np.ndarray,
    x_hat: np.ndarray,
) -> DescriptiveResult:
    """Compute signal-to-quantization-noise ratio in dB.

    SQNR = 10 * log10(||x||^2 / ||x - x_hat||^2)

    :param x: Original signal.
    :param x_hat: Quantized (reconstructed) signal.
    :return: DescriptiveResult with SQNR in dB.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x_hat = np.asarray(x_hat, dtype=np.float64).ravel()
    if len(x) != len(x_hat):
        raise ValueError("x and x_hat must have the same length")
    signal_power = float(np.sum(x**2))
    noise_power = float(np.sum((x - x_hat) ** 2))
    if noise_power < 1e-30:
        sqnr_db = float("inf")
    elif signal_power < 1e-30:
        sqnr_db = 0.0
    else:
        sqnr_db = 10.0 * np.log10(signal_power / noise_power)
    return DescriptiveResult(
        name="sqnr",
        value=sqnr_db,
        extra={
            "signal_power": signal_power,
            "noise_power": noise_power,
            "sqnr_db": sqnr_db,
            "mse": noise_power / len(x),
        },
    )


def cheatsheet() -> str:
    return "signal_quant_noise_ratio(x, x_hat) -> SQNR in dB"


sqnr = signal_quant_noise_ratio
