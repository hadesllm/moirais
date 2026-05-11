# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Activation-aware weight quantization (AWQ)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def activation_aware_quant(
    W: np.ndarray,
    activations: np.ndarray,
    bits: int = 4,
) -> DescriptiveResult:
    """Activation-aware weight quantization.

    Scales weight columns by activation magnitude before quantization
    to protect salient weights, then unscales after.

    :param W: Weight matrix (rows x cols).
    :param activations: Activation statistics per input channel (1-D, length = cols).
    :param bits: Quantization bit width.
    :return: DescriptiveResult with AWQ-quantized weights and MSE.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    activations = np.asarray(activations, dtype=np.float64).ravel()
    if len(activations) != W.shape[1]:
        raise ValueError("activations length must match W columns")
    act_scale = np.clip(activations / (np.max(activations) + 1e-12), 0.01, 1.0)
    scale_factor = 1.0 / act_scale
    W_scaled = W * scale_factor[None, :]
    vmin, vmax = float(W_scaled.min()), float(W_scaled.max())
    levels = 2**bits - 1
    rng = vmax - vmin if vmax > vmin else 1.0
    scaled = (W_scaled - vmin) / rng * levels
    codes = np.clip(np.round(scaled), 0, levels)
    W_q_scaled = codes / levels * rng + vmin
    W_q = W_q_scaled * act_scale[None, :]
    mse = float(np.mean((W - W_q) ** 2))
    return DescriptiveResult(
        name="activation_aware_quant",
        value=mse,
        extra={
            "W_quantized": W_q,
            "scale_factor": scale_factor,
            "bits": bits,
            "mse": mse,
        },
    )


def cheatsheet() -> str:
    return "activation_aware_quant(W, activations, bits) -> AWQ"


awq = activation_aware_quant
