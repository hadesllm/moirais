"""TurboQuant MSE-optimal quantization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def turboquant_mse(
    x: np.ndarray,
    bits: int = 3,
) -> DescriptiveResult:
    """TurboQuant MSE-optimal scalar quantization.

    Applies Walsh-Hadamard randomization followed by Lloyd-Max
    MSE-optimal codebook quantization at the given bit width.

    :param x: Input vector (1-D float array).
    :param bits: Quantization bit width (2, 3, or 4).
    :return: DescriptiveResult with compressed block.
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
    levels = 2**bits
    vmin, vmax = float(h.min()), float(h.max())
    if vmax == vmin:
        codes = np.zeros(n, dtype=np.int32)
        centroids = np.array([vmin])
    else:
        edges = np.linspace(vmin, vmax, levels + 1)
        codes = np.clip(np.digitize(h, edges[1:-1]), 0, levels - 1)
        centroids = np.array([h[codes == c].mean() if np.any(codes == c) else edges[c] for c in range(levels)])
        for _ in range(20):
            new_centroids = np.array(
                [h[codes == c].mean() if np.any(codes == c) else centroids[c] for c in range(levels)]
            )
            if np.allclose(centroids, new_centroids, atol=1e-12):
                break
            centroids = new_centroids
            dists = np.abs(h[:, None] - centroids[None, :])
            codes = np.argmin(dists, axis=1).astype(np.int32)
    h_hat = centroids[codes]
    mse = float(np.mean((h - h_hat) ** 2))
    ratio = 32.0 / bits
    return DescriptiveResult(
        name="turboquant_mse",
        value=mse,
        extra={
            "codes": codes[:d],
            "centroids": centroids,
            "scale": scale,
            "bits": bits,
            "n_original": d,
            "n_padded": n,
            "compression_ratio": ratio,
            "h_transformed": h[:d],
            "h_hat": h_hat[:d],
        },
    )


def cheatsheet() -> str:
    return "turboquant_mse(x, bits=3) -> TurboQuant MSE-optimal quantization"


tqmse = turboquant_mse
