"""TurboQuant product quantizer."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def turboquant_prod(
    x: np.ndarray,
    bits: int = 3,
    n_sub: int = 2,
) -> DescriptiveResult:
    """TurboQuant product quantizer.

    Splits the WHT-transformed vector into *n_sub* sub-vectors and
    independently quantizes each with a per-sub codebook.

    :param x: Input vector (1-D float array).
    :param bits: Bits per sub-vector entry.
    :param n_sub: Number of sub-vector partitions.
    :return: DescriptiveResult with product-quantized block.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    d = len(x)
    n = int(2 ** np.ceil(np.log2(max(d, 2))))
    if n > d:
        x = np.concatenate([x, np.zeros(n - d)])
    scale = 1.0 / np.sqrt(n)
    h = x.copy()
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                a, b = h[i + j], h[i + j + step]
                h[i + j] = a + b
                h[i + j + step] = a - b
        step *= 2
    h *= scale
    sub_len = n // n_sub
    levels = 2**bits
    all_codes = []
    all_centroids = []
    total_mse = 0.0
    for s in range(n_sub):
        sub = h[s * sub_len : (s + 1) * sub_len]
        vmin, vmax = float(sub.min()), float(sub.max())
        if vmax == vmin:
            codes_s = np.zeros(sub_len, dtype=np.int32)
            cents = np.array([vmin])
        else:
            edges = np.linspace(vmin, vmax, levels + 1)
            codes_s = np.clip(np.digitize(sub, edges[1:-1]), 0, levels - 1)
            cents = np.array([sub[codes_s == c].mean() if np.any(codes_s == c) else edges[c] for c in range(levels)])
        all_codes.append(codes_s)
        all_centroids.append(cents)
        total_mse += float(np.mean((sub - cents[codes_s]) ** 2))
    mse = total_mse / n_sub
    return DescriptiveResult(
        name="turboquant_prod",
        value=mse,
        extra={
            "codes": [c[: d // n_sub] for c in all_codes],
            "centroids": all_centroids,
            "bits": bits,
            "n_sub": n_sub,
            "compression_ratio": 32.0 / bits,
        },
    )


def cheatsheet() -> str:
    return "turboquant_prod(x, bits=3, n_sub=2) -> TurboQuant product quantizer"


tqprd = turboquant_prod
