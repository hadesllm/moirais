"""Vector quantization with nearest codeword."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def vector_quantize(
    x: np.ndarray,
    codebook: np.ndarray,
) -> DescriptiveResult:
    """Vector quantization: assign each vector to nearest codeword.

    :param x: Input vectors (n x dim) or (dim,).
    :param codebook: Codebook (k x dim).
    :return: DescriptiveResult with codes and reconstruction MSE.
    """
    x = np.asarray(x, dtype=np.float64)
    codebook = np.asarray(codebook, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if codebook.ndim == 1:
        codebook = codebook.reshape(-1, 1)
    dists = np.linalg.norm(x[:, None, :] - codebook[None, :, :], axis=2)
    codes = np.argmin(dists, axis=1)
    x_hat = codebook[codes]
    mse = float(np.mean((x - x_hat) ** 2))
    return DescriptiveResult(
        name="vector_quantize",
        value=mse,
        extra={
            "codes": codes,
            "reconstructed": x_hat.squeeze(),
            "mse": mse,
            "k": codebook.shape[0],
        },
    )


def cheatsheet() -> str:
    return "vector_quantize(x, codebook) -> VQ with nearest codeword"


vq = vector_quantize
