# moirais.fn — function file (hadesllm/moirais)
"""Dequantization error metrics (MSE, cosine, SQNR)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dequant_error(
    original: np.ndarray,
    quantized: np.ndarray,
) -> DescriptiveResult:
    """Compute dequantization error metrics: MSE, cosine similarity, SQNR.

    :param original: Original float vector.
    :param quantized: Quantized/reconstructed vector.
    :return: DescriptiveResult with MSE, cosine similarity, and SQNR.
    """
    original = np.asarray(original, dtype=np.float64).ravel()
    quantized = np.asarray(quantized, dtype=np.float64).ravel()
    if len(original) != len(quantized):
        raise ValueError("original and quantized must have the same length")
    mse = float(np.mean((original - quantized) ** 2))
    no = np.linalg.norm(original)
    nq = np.linalg.norm(quantized)
    if no > 0 and nq > 0:
        cosine = float(np.dot(original, quantized) / (no * nq))
    else:
        cosine = 1.0 if no == 0 and nq == 0 else 0.0
    signal_power = float(np.sum(original**2))
    noise_power = float(np.sum((original - quantized) ** 2))
    if noise_power < 1e-30:
        sqnr_db = float("inf")
    elif signal_power < 1e-30:
        sqnr_db = 0.0
    else:
        sqnr_db = 10.0 * np.log10(signal_power / noise_power)
    max_err = float(np.max(np.abs(original - quantized)))
    return DescriptiveResult(
        name="dequant_error",
        value=mse,
        extra={
            "mse": mse,
            "cosine_similarity": cosine,
            "sqnr_db": sqnr_db,
            "max_absolute_error": max_err,
            "rmse": float(np.sqrt(mse)),
        },
    )


def cheatsheet() -> str:
    return "dequant_error(original, quantized) -> MSE/cosine/SQNR error metrics"


dqerr = dequant_error
