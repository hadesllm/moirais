"""TurboQuant dequantization (inverse)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def turboquant_decode(
    codes: np.ndarray,
    centroids: np.ndarray,
    n_original: int,
    bits: int = 3,
) -> DescriptiveResult:
    """Dequantize a TurboQuant block back to float vector.

    Reconstructs from codebook indices via inverse WHT.
    Uses 1/sqrt(d) normalization (orthonormal WHT).

    :param codes: Integer code indices from quantization.
    :param centroids: Codebook centroids.
    :param n_original: Original vector length before padding.
    :param bits: Bit width used during quantization.
    :return: DescriptiveResult with reconstructed vector.
    """
    codes = np.asarray(codes, dtype=np.int32).ravel()
    centroids = np.asarray(centroids, dtype=np.float64)
    n = len(codes)
    n_full = int(2 ** np.ceil(np.log2(max(n, 2))))
    if n_full > n:
        codes = np.concatenate([codes, np.zeros(n_full - n, dtype=np.int32)])
    h_hat = centroids[codes]
    scale = 1.0 / np.sqrt(n_full)
    x_hat = h_hat.copy()
    step = 1
    while step < n_full:
        for i in range(0, n_full, step * 2):
            for j in range(step):
                a, b = x_hat[i + j], x_hat[i + j + step]
                x_hat[i + j] = a + b
                x_hat[i + j + step] = a - b
        step *= 2
    x_hat *= scale
    return DescriptiveResult(
        name="turboquant_decode",
        value=float(np.mean(np.abs(x_hat[:n_original]))),
        extra={
            "reconstructed": x_hat[:n_original],
            "bits": bits,
            "n_original": n_original,
        },
    )


def cheatsheet() -> str:
    return "turboquant_decode(codes, centroids, n_original) -> dequantize TQ block"


tqdec = turboquant_decode
