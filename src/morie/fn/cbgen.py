# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Codebook generation via k-means for vector quantization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def codebook_generate(
    data: np.ndarray,
    k: int = 256,
    max_iter: int = 50,
    seed: int = 42,
) -> DescriptiveResult:
    """Generate a VQ codebook using k-means clustering.

    :param data: Training vectors (n_samples x dim) or 1-D.
    :param k: Number of codewords.
    :param max_iter: Maximum k-means iterations.
    :param seed: Random seed.
    :return: DescriptiveResult with codebook and distortion.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, dim = data.shape
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    codebook = data[idx].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - codebook[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_codebook = np.array(
            [data[labels == c].mean(axis=0) if np.any(labels == c) else codebook[c] for c in range(k)]
        )
        if np.allclose(codebook, new_codebook, atol=1e-10):
            codebook = new_codebook
            break
        codebook = new_codebook
    dists = np.linalg.norm(data[:, None, :] - codebook[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    distortion = float(np.mean(np.min(dists, axis=1) ** 2))
    return DescriptiveResult(
        name="codebook_generate",
        value=distortion,
        extra={
            "codebook": codebook,
            "labels": labels,
            "k": k,
            "dim": dim,
            "distortion": distortion,
        },
    )


def cheatsheet() -> str:
    return "codebook_generate(data, k) -> k-means codebook for VQ"


cbgen = codebook_generate
