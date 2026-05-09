# moirais.fn — function file (hadesllm/moirais)
"""GPTQ Hessian-based weight quantizer."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gptq_quantize(
    W: np.ndarray,
    H: np.ndarray | None = None,
    bits: int = 4,
) -> DescriptiveResult:
    """GPTQ-style weight quantization using Hessian information.

    Quantizes weight matrix W column-by-column, using the inverse
    Hessian to compensate remaining columns for quantization error.

    :param W: Weight matrix (rows x cols).
    :param H: Hessian approximation (cols x cols). If None, uses identity.
    :param bits: Quantization bit width.
    :return: DescriptiveResult with quantized weights and MSE.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    rows, cols = W.shape
    if H is None:
        H = np.eye(cols)
    else:
        H = np.asarray(H, dtype=np.float64)
    damp = 0.01 * np.mean(np.diag(H))
    H_inv = np.linalg.inv(H + damp * np.eye(cols))
    W_q = W.copy()
    levels = 2**bits - 1
    vmin, vmax = float(W.min()), float(W.max())
    rng = vmax - vmin if vmax > vmin else 1.0
    for col in range(cols):
        w_col = W_q[:, col]
        scaled = (w_col - vmin) / rng * levels
        q_col = np.clip(np.round(scaled), 0, levels) / levels * rng + vmin
        error = w_col - q_col
        W_q[:, col] = q_col
        if col + 1 < cols:
            W_q[:, col + 1 :] += np.outer(error, H_inv[col, col + 1 :]) / (H_inv[col, col] + 1e-12)
    mse = float(np.mean((W - W_q) ** 2))
    return DescriptiveResult(
        name="gptq_quantize",
        value=mse,
        extra={
            "W_quantized": W_q,
            "bits": bits,
            "shape": W.shape,
            "mse": mse,
        },
    )


def cheatsheet() -> str:
    return "gptq_quantize(W, H, bits) -> GPTQ weight quantizer (Hessian-based)"


gptq = gptq_quantize
